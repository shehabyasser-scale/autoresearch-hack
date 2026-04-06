"""
Autonomous Atari autoresearch runner.

Runs the full experiment loop: asks an LLM to modify agent.py,
evaluates the result, keeps improvements, discards failures, repeats.

Usage:
    # Default: uses ANTHROPIC_AUTH_TOKEN env var + Scale litellm proxy
    uv run run.py

    # Explicit key
    uv run run.py --api-key <token>

    # Direct Anthropic API (no proxy)
    uv run run.py --api-key sk-ant-... --api-base https://api.anthropic.com

    # Customize
    uv run run.py --model anthropic/claude-opus-4-6 --max-experiments 50
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# Suppress litellm's noisy logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
os.environ["LITELLM_LOG"] = "ERROR"

ATARI_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ATARI_DIR)
AGENT_FILE = os.path.join(ATARI_DIR, "agent.py")
PREPARE_FILE = os.path.join(ATARI_DIR, "prepare.py")
RESULTS_FILE = os.path.join(ATARI_DIR, "results.tsv")
RUN_LOG = os.path.join(ATARI_DIR, "run.log")
VIDEOS_DIR = os.path.join(ATARI_DIR, "videos")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args):
    """Run a git command in the repo directory, return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_short_hash():
    return git("rev-parse", "--short", "HEAD")


def git_commit(message):
    git("add", AGENT_FILE)
    git("commit", "-m", message)
    return git_short_hash()


def git_reset_hard(commit):
    git("reset", "--hard", commit)


def git_checkout_branch(branch):
    """Create or switch to branch. If it already exists, just switch."""
    result = subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        if "already exists" in result.stderr:
            subprocess.run(
                ["git", "checkout", branch],
                cwd=REPO_DIR, capture_output=True, text=True,
            )
        else:
            print(f"  git checkout -b {branch} failed: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    """Run agent.py and return (mean_reward, raw_output) or (None, error_output) on crash."""
    result = subprocess.run(
        [sys.executable, "agent.py"],
        cwd=ATARI_DIR, capture_output=True, text=True, timeout=4500,
    )

    output = result.stdout + result.stderr

    # Write run.log
    with open(RUN_LOG, "w") as f:
        f.write(output)

    if result.returncode != 0:
        return None, output

    match = re.search(r"^mean_reward:\s+([\d.]+)", output, re.MULTILINE)
    if not match:
        return None, output

    return float(match.group(1)), output


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

def record_video(experiment_num, mean_reward, num_episodes=1):
    """Record a short video of the current agent playing, saved to videos/."""
    try:
        import gymnasium as gym
        import ale_py  # noqa: F401
        import importlib
        import importlib.util

        os.makedirs(VIDEOS_DIR, exist_ok=True)

        # Import agent module fresh (it may have been rewritten)
        spec = importlib.util.spec_from_file_location("agent_mod", AGENT_FILE)
        agent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_mod)

        env = gym.make(
            "ALE/Breakout-v5",
            render_mode="rgb_array",
            frameskip=1,
            repeat_action_probability=0.25,
        )

        prefix = f"exp{experiment_num:04d}_reward{mean_reward:.0f}"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEOS_DIR,
            name_prefix=prefix,
            episode_trigger=lambda ep: ep < num_episodes,
        )

        agent = agent_mod.Agent(env.action_space)

        for ep in range(num_episodes):
            obs, info = env.reset(seed=42 + ep)
            if hasattr(agent, "reset"):
                agent.reset()

            steps = 0
            while True:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated or steps >= 10_000:
                    break

        env.close()
        print(f"[video] Saved to {VIDEOS_DIR}/{prefix}*.mp4")

    except Exception as e:
        print(f"[video] Recording failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

def init_results():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("commit\tmean_reward\tstatus\tdescription\terror\n")


def log_result(commit, mean_reward, status, description, error=""):
    # Sanitize error for TSV (collapse whitespace, remove tabs/newlines)
    error_clean = " ".join(error.split())[:200] if error else ""
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{mean_reward:.4f}\t{status}\t{description}\t{error_clean}\n")


# ---------------------------------------------------------------------------
# Rate limit handling
# ---------------------------------------------------------------------------

def parse_rate_limit_reset(error_str):
    """Parse 'resets at: YYYY-MM-DD HH:MM:SS UTC' and return seconds to wait, or None."""
    match = re.search(r"resets at:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*UTC", error_str)
    if not match:
        return None
    try:
        reset_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        wait = (reset_time - datetime.now(timezone.utc)).total_seconds() + 2
        return max(wait, 5)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous AI researcher optimizing a DQN agent for Atari Breakout.

Your goal: maximize mean_reward on a fixed 30-episode evaluation.

You modify agent.py -- the only file you can edit. The evaluation harness \
(prepare.py) is fixed and cannot be changed.

The agent uses a Deep Q-Network (DQN) architecture:
- CNN processes 4 stacked 84x84 grayscale frames
- train() trains the DQN within a 3600s (1 hour) time budget using a replay buffer
- Agent.act() uses the trained network for greedy action selection at eval time

Available imports: numpy, time, random, collections.deque, torch, torch.nn, \
torch.optim, and from prepare.py: GAME, TIME_BUDGET, make_env, evaluate_agent.

Constraints:
- ALL experiments MUST use DQN (deep Q-learning with neural networks)
- The Agent class must implement act(obs) -> int and reset() -> None
- act() receives a (210, 160, 3) uint8 RGB observation
- Breakout actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
- train() MUST actually train the network for TIME_BUDGET seconds (3600s = 1 hour)
- Do NOT replace DQN with heuristics -- improve the DQN itself

Things to experiment with:
- Network architecture (deeper CNN, dueling DQN, noisy nets)
- Double DQN (use policy net to select actions, target net to evaluate)
- Prioritized experience replay
- Reward clipping, frame skipping, observation preprocessing
- Hyperparameter tuning (LR, batch size, epsilon schedule, gamma, buffer size)
- N-step returns, gradient clipping strategies

When asked to suggest a modification, respond with ONLY the complete new \
agent.py file between ```python and ``` markers. No other text."""


def build_prompt(agent_code, results_history, best_reward, attempt_num):
    """Build the prompt for the LLM."""
    # Extract recent crash errors to help the LLM avoid repeating them
    crash_warnings = ""
    if results_history:
        recent_crashes = []
        for line in results_history.strip().split("\n")[-20:]:  # last 20 experiments
            parts = line.split("\t")
            if len(parts) >= 5 and parts[2] == "crash" and parts[4].strip():
                recent_crashes.append(parts[4].strip())
        if recent_crashes:
            # Deduplicate
            unique_crashes = list(dict.fromkeys(recent_crashes))[-5:]
            crash_warnings = "\n\nRECENT CRASH ERRORS (avoid these!):\n"
            for err in unique_crashes:
                crash_warnings += f"  - {err}\n"
            crash_warnings += "\nCommon fixes: use integer slices (not float), ensure obs is a numpy array \
(not tuple) before indexing, verify tensor shapes match between layers.\n"

    prompt = f"""Here is the current agent.py (best so far, mean_reward={best_reward:.4f}):

```python
{agent_code}
```

Experiment history (most recent last):
{results_history if results_history else "(no experiments yet — this will be the first modification)"}
{crash_warnings}
This is experiment #{attempt_num}. Suggest a modification to agent.py that will \
improve mean_reward. Think about what has worked and what hasn't based on the \
history. Try something different from previous failed attempts.

IMPORTANT: Make sure your code actually runs without errors. Test edge cases \
mentally: observation shapes, integer vs float slicing, tensor dimensions.

Respond with the complete new agent.py file between ```python and ``` markers."""
    return prompt


def call_llm(api_key, model, prompt, api_base=None):
    """Call the LLM via litellm. Retries are handled by the caller."""
    from litellm import completion

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.7,
        api_key=api_key,
        api_base=api_base,
    )
    return response.choices[0].message.content


def call_llm_with_retry(api_key, model, prompt, api_base=None, max_retries=10):
    """Call LLM with smart retry: parse rate limit reset time, backoff on other errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return call_llm(api_key, model, prompt, api_base)
        except Exception as e:
            error_str = str(e)
            wait = parse_rate_limit_reset(error_str)
            if wait is not None:
                print(f"[llm] Rate limited. Waiting {wait:.0f}s until reset... (attempt {attempt}/{max_retries})")
                time.sleep(wait)
            else:
                backoff = min(30 * attempt, 300)
                print(f"[llm] Error: {e}")
                print(f"[llm] Retrying in {backoff}s... (attempt {attempt}/{max_retries})")
                time.sleep(backoff)

    return None  # all retries exhausted


def extract_code(response):
    """Extract Python code from markdown code block in LLM response."""
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try without language tag
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Atari autoresearch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Uses ANTHROPIC_AUTH_TOKEN env var + Scale litellm proxy by default
  uv run run.py

  # Explicit key
  uv run run.py --api-key <token>

  # Direct Anthropic API (no proxy)
  uv run run.py --api-key sk-ant-... --api-base https://api.anthropic.com

  # Custom model
  uv run run.py --model anthropic/claude-opus-4-6
        """,
    )
    parser.add_argument("--api-key",
                        default=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("LITELLM_API_KEY"),
                        help="API key (or set ANTHROPIC_AUTH_TOKEN / LITELLM_API_KEY env var)")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514",
                        help="LiteLLM model name (default: anthropic/claude-sonnet-4-20250514)")
    parser.add_argument("--api-base",
                        default=os.environ.get("LITELLM_API_BASE", "https://litellm-proxy.ml-serving-internal.scale.com"),
                        help="API base URL (default: Scale litellm proxy, or set LITELLM_API_BASE env var)")
    parser.add_argument("--tag", default=None,
                        help="Branch tag (default: today's date, e.g. apr4)")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Max experiments to run (0 = unlimited)")
    parser.add_argument("--no-git", action="store_true",
                        help="Skip git operations (useful for testing)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key required (or set ANTHROPIC_AUTH_TOKEN env var)")
        sys.exit(1)

    # --- Setup ---
    tag = args.tag or datetime.now().strftime("%b%d").lower()
    branch = f"atari-research/{tag}"

    print("=" * 60)
    print("  ATARI AUTORESEARCH")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Branch:  {branch}")
    print(f"  Limit:   {'unlimited' if args.max_experiments == 0 else args.max_experiments}")
    print("=" * 60)
    print()

    if not args.no_git:
        git_checkout_branch(branch)

    init_results()

    # --- Baseline ---
    print("[baseline] Running baseline evaluation...")
    with open(AGENT_FILE) as f:
        best_code = f.read()

    mean_reward, output = run_evaluation()
    if mean_reward is None:
        print("[baseline] CRASHED! Fix agent.py and retry.")
        print(output[-500:])
        sys.exit(1)

    best_reward = mean_reward
    baseline_commit = git_short_hash()
    log_result(baseline_commit, best_reward, "keep", "baseline heuristic")
    print(f"[baseline] mean_reward={best_reward:.4f}")

    # Record baseline video
    print(f"[video] Recording baseline gameplay...")
    record_video(0, best_reward)
    print()

    # --- Experiment loop ---
    experiment = 0
    kept = 0
    discarded = 0

    while True:
        experiment += 1
        if args.max_experiments > 0 and experiment > args.max_experiments:
            break

        print(f"{'='*60}")
        print(f"  EXPERIMENT {experiment}")
        print(f"  Best so far: {best_reward:.4f} | Kept: {kept} | Discarded: {discarded}")
        print(f"{'='*60}")

        exp_start = time.time()

        # Read current results history
        results_history = ""
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                results_history = f.read()

        # Ask LLM for a modification
        print("[llm] Asking for modification...")
        llm_start = time.time()
        prompt = build_prompt(best_code, results_history, best_reward, experiment)
        response = call_llm_with_retry(args.api_key, args.model, prompt, args.api_base)

        if response is None:
            print("[llm] All retries exhausted. Skipping experiment.")
            continue

        llm_elapsed = time.time() - llm_start
        print(f"[llm] Response received in {llm_elapsed:.0f}s")

        new_code = extract_code(response)
        if not new_code:
            print("[llm] Failed to extract code from response. Skipping.")
            continue

        # Extract description from the agent class docstring or first comment
        desc_match = re.search(r"#\s*(.+)", new_code.split("class Agent")[0][-200:]) if "class Agent" in new_code else None
        description = desc_match.group(1).strip() if desc_match else f"experiment {experiment}"
        description = description[:80]

        # Write new agent.py
        with open(AGENT_FILE, "w") as f:
            f.write(new_code)

        # Commit
        pre_commit = git_short_hash()
        if not args.no_git:
            commit_hash = git_commit(f"try: {description}")
        else:
            commit_hash = "nogit"

        # Evaluate
        code_lines = len(new_code.splitlines())
        has_torch = "import torch" in new_code
        has_dqn = any(kw in new_code.lower() for kw in ["dqn", "qnetwork", "q_network", "replay"])
        print(f"[eval] Running: {description}")
        print(f"[eval] Code: {code_lines} lines | torch={'yes' if has_torch else 'no'} | dqn={'yes' if has_dqn else 'no'}")
        eval_start = time.time()
        try:
            mean_reward, output = run_evaluation()
        except subprocess.TimeoutExpired:
            mean_reward = None
            output = "TIMEOUT: evaluation exceeded 75 minutes"

        eval_elapsed = time.time() - eval_start

        if mean_reward is None:
            # Crash — extract the actual error line
            error_tail = output[-500:] if output else "no output"
            # Find the last exception line for a clean error message
            error_lines = [l.strip() for l in error_tail.splitlines() if l.strip()]
            error_msg = error_lines[-1] if error_lines else "unknown error"

            print(f"[eval] CRASHED after {eval_elapsed:.0f}s")
            print(f"[eval] Error: {error_msg}")
            print(f"  {error_tail[-200:]}")
            log_result(commit_hash, 0.0, "crash", description, error_msg)
            if not args.no_git:
                git_reset_hard(pre_commit)
            else:
                with open(AGENT_FILE, "w") as f:
                    f.write(best_code)
            discarded += 1

        elif mean_reward > best_reward:
            # Improvement!
            print(f"[eval] IMPROVED: {best_reward:.4f} -> {mean_reward:.4f} (+{mean_reward - best_reward:.4f}) in {eval_elapsed:.0f}s")
            log_result(commit_hash, mean_reward, "keep", description)
            best_reward = mean_reward
            best_code = new_code
            kept += 1

            # Record a video of the improved agent
            print(f"[video] Recording gameplay...")
            record_video(experiment, mean_reward)

        else:
            # No improvement
            print(f"[eval] No improvement: {mean_reward:.4f} <= {best_reward:.4f} (ran {eval_elapsed:.0f}s)")
            log_result(commit_hash, mean_reward, "discard", description)
            if not args.no_git:
                git_reset_hard(pre_commit)
            else:
                with open(AGENT_FILE, "w") as f:
                    f.write(best_code)
            discarded += 1

        print()

    # --- Summary ---
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Experiments: {experiment - 1}")
    print(f"  Kept:        {kept}")
    print(f"  Discarded:   {discarded}")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Results:     {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
