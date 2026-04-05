# Atari AutoResearch

> Autonomous AI agent optimization for Atari Breakout — an LLM iteratively modifies the agent code, evaluates it, and keeps only improvements.

---

## Quick Start

### Prerequisites

- Python 3.9-3.12
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for Claude (via Anthropic or Scale litellm proxy)

### Setup

```bash
cd atari
uv sync                              # creates .venv + installs all dependencies
uv run prepare.py                    # verify environment works (quick sanity check)
```

### Run Overnight

```bash
# Default: uses ANTHROPIC_AUTH_TOKEN env var + Scale litellm proxy
uv run run.py --tag apr05

# With explicit API key
uv run run.py --api-key <YOUR_KEY> --tag apr05

# Direct Anthropic API (no proxy)
uv run run.py --api-key sk-ant-... --api-base https://api.anthropic.com --tag apr05

# Limit number of experiments
uv run run.py --tag apr05 --max-experiments 10

# Custom model
uv run run.py --tag apr05 --model anthropic/claude-opus-4-6
```

The `--tag` creates a git branch like `atari-research/apr05` to isolate this run's commits.

### After the Run

```bash
# Generate a 6-panel analysis chart
uv run analyze.py

# View the chart
open run_analysis.png

# Watch the gameplay videos (one per kept experiment)
ls videos/

# Check the raw results log
cat results.tsv
```

---

## How It Works

1. **Baseline** — evaluate the current `agent.py` as-is
2. **Loop** (runs forever until Ctrl+C or `--max-experiments`):
   - Read experiment history from `results.tsv`
   - Ask Claude to modify `agent.py` (with full history context)
   - Write the new code, `git commit`
   - Run the agent: `train()` for up to **1 hour**, then 30 evaluation episodes
   - If `mean_reward` improved: **keep** (new baseline) + record gameplay video
   - If not: **discard** (`git reset --hard` to previous)
   - If crashed: **log crash** + rollback

---

## Timing

Each experiment takes approximately **65-75 minutes**:

| Phase | Time |
|-------|------|
| LLM API call | ~5-30s |
| `train()` | **up to 3600s (1 hour)** |
| 30 evaluation episodes | ~1-5 min |
| Video recording (if kept) | ~10-30s |
| Git + logging | ~2s |

Overnight (12 hours) expect **~10-11 experiments**.

---

## What a Run Produces

| File | Description | Persists? |
|------|-------------|-----------|
| `results.tsv` | Every experiment: commit, reward, status, description | Appends across runs |
| `run.log` | stdout+stderr of the **last** evaluation only | Overwritten each experiment |
| `agent.py` | The current best agent code (on the research branch) | Via git |
| `videos/` | `.mp4` gameplay recordings of each **kept** experiment | Accumulates |
| Git history | Each "keep" is a surviving commit on the research branch | Yes |

### When Does an Experiment End?

An experiment ends when **any** of these conditions is met:

1. **Evaluation completes normally** — `agent.py` runs `train()` (up to 3600s / 1 hour), then plays 30 episodes (up to 10,000 steps each). The process exits, `mean_reward` is parsed from stdout.
2. **The subprocess crashes** — any Python error (syntax, runtime, OOM). Non-zero exit code -> logged as "crash".
3. **Timeout (75 minutes)** — if the subprocess hasn't exited after 4500s, it's killed. Logged as "crash".

The **entire loop** (`run.py`) runs forever unless:
- `--max-experiments N` is set and reached
- You kill the process (Ctrl+C)
- All LLM retries are permanently exhausted (unlikely)

### Console Output

```
============================================================
  EXPERIMENT 42
  Best so far: 13.6000 | Kept: 8 | Discarded: 25
============================================================
[llm] Asking for modification...
[eval] Running: dueling DQN with prioritized replay
[eval] IMPROVED: 13.6000 -> 14.3333 (+0.7333)
[video] Recording gameplay...
[video] Saved to videos/exp0042_reward14*.mp4
```

---

## CLI Reference

```
uv run run.py [OPTIONS]

Options:
  --api-key KEY          API key (or set ANTHROPIC_AUTH_TOKEN env var)
  --model MODEL          LiteLLM model name (default: anthropic/claude-sonnet-4-20250514)
  --api-base URL         API base URL (default: Scale litellm proxy)
  --tag TAG              Branch tag (default: today's date, e.g. apr05)
  --max-experiments N    Max experiments to run (0 = unlimited, default)
  --no-git               Skip git operations (useful for testing)
```

---

## Videos

Every time an experiment **improves** the best reward, a gameplay video is automatically recorded to `atari/videos/`. Videos are named like:

```
videos/exp0000_reward3-episode-0.mp4      # baseline
videos/exp0005_reward7-episode-0.mp4      # first improvement
videos/exp0008_reward186-episode-0.mp4    # DQN breakthrough
```

To manually record a video of the current agent:

```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from prepare import GAME, SEED
from agent import Agent

env = gym.make(GAME, render_mode="rgb_array", frameskip=1)
env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda e: True)
agent = Agent(env.action_space)
obs, _ = env.reset(seed=SEED)
agent.reset()
done = False
while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()
```

---

## Analyzing Results

Run the built-in analyzer:

```bash
uv run analyze.py              # generates run_analysis.png + prints summary
uv run analyze.py --no-show    # save PNG only, don't open window
```

This produces a 6-panel chart:
1. All experiments scatter (color-coded by keep/discard/crash)
2. Reward progression (kept experiments only, monotonic)
3. Outcome distribution (bar chart with percentages)
4. Crash rate over time (rolling window)
5. Reward distribution histogram (non-crash experiments)
6. Cumulative best reward

### results.tsv Format

```
commit    mean_reward    status    description
c45a82e   3.0667         keep      baseline heuristic
2ac7fda   7.1333         keep      improved ball tracking
fdd8bfc   3.0667         discard   worse architecture
8459ed1   0.0000         crash     syntax error
```

---

## Files Reference

| File | Role | Modified by |
|------|------|-------------|
| `prepare.py` | Environment factory + evaluation harness (30 episodes, fixed seed, 1hr budget) | Nobody (frozen) |
| `agent.py` | The agent: `act(obs) -> action`, `reset()`, `train()` | LLM via run.py |
| `run.py` | Autonomous orchestrator: LLM calls, git ops, result logging, video recording | Nobody (frozen) |
| `program.md` | Instructions that guide the LLM researcher | Human |
| `analyze.py` | Results visualization: generates 6-panel chart from results.tsv | Nobody (utility) |
| `results.tsv` | Append-only experiment log | run.py |
| `run.log` | Last experiment's stdout/stderr | run.py (overwritten) |
| `videos/` | Gameplay recordings of kept experiments | run.py |

---

---

# Architecture & System Design

> A visual deep-dive into how the autonomous AI research platform works — from high-level orchestration down to individual components.

```
    LLM proposes change  ──►  System evaluates  ──►  Only improvements survive
         ▲                                                      │
         └──────────────── History fed back ◄───────────────────┘
```

---

## Diagram Index

| # | Section | Diagram Types |
|---|---------|---------------|
| 1 | [System Context](#1-system-context) | C4 Context |
| 2 | [Project Anatomy](#2-project-anatomy) | Mindmap |
| 3 | [The Experiment Loop](#3-the-experiment-loop) | Flowchart |
| 4 | [Single Experiment Lifecycle](#4-single-experiment-lifecycle) | Sequence |
| 5 | [Experiment State Machine](#5-experiment-state-machine) | State Diagram |
| 6 | [Developer Journey](#6-developer-journey) | Journey Map |
| 7 | [Agent Class Design](#7-agent-class-design) | Class Diagram |
| 8 | [Observation Pipeline](#8-observation-pipeline) | Data Flow |
| 9 | [Ball Detection Algorithm](#9-ball-detection-algorithm) | Decision Flowchart |
| 10 | [Trajectory Prediction](#10-trajectory-prediction) | Flowchart |
| 11 | [Evaluation Harness](#11-evaluation-harness) | Flowchart |
| 12 | [LLM Prompt Protocol](#12-llm-prompt-protocol) | Block Diagram |
| 13 | [Git Branching Strategy](#13-git-branching-strategy) | Git Graph |
| 14 | [Experiment Timeline](#14-experiment-timeline) | Gantt |
| 15 | [Data Model](#15-data-model) | ER Diagram |
| 16 | [Action Space](#16-action-space) | Pie + Block |
| 17 | [Error Handling & Recovery](#17-error-handling--recovery) | Flowchart |
| 18 | [Dual-Track Comparison](#18-dual-track-comparison) | Side-by-Side |
| 19 | [Dependency Graph](#19-dependency-graph) | Flowchart |
| 20 | [Outcome Space](#20-outcome-space) | Quadrant Chart |

**Color Legend** used consistently across all diagrams:

| Color | Meaning |
|-------|---------|
| Green | Success / kept / start |
| Red | Failure / crash / error |
| Orange | Warning / discard |
| Purple | LLM / AI interaction |
| Blue | Internal components |
| Dark | External services |

---

## 1. System Context

The 10-second overview. AutoResearch sits between an LLM and a game environment, orchestrating autonomous experiments.

```mermaid
graph TB
    USER(("Researcher<br/><i>Starts the runner,<br/>reviews results</i>"))

    subgraph AUTORESEARCH["AutoResearch Platform"]
        direction LR
        RUNNER["run.py<br/>Autonomous Orchestrator"]
        AGENT["agent.py<br/>Modifiable Agent"]
        HARNESS["prepare.py<br/>Fixed Eval Harness"]
    end

    LLM["Claude API<br/><i>Generates code<br/>modifications</i>"]
    ALE["Atari Learning<br/>Environment<br/><i>ALE/Breakout-v5</i>"]
    GIT["Git Repository<br/><i>Version control<br/>& rollback</i>"]
    LOG["results.tsv<br/><i>Experiment<br/>provenance</i>"]

    USER -->|"uv run run.py"| RUNNER
    RUNNER <-->|"prompt / code"| LLM
    RUNNER -->|"writes"| AGENT
    AGENT -->|"evaluated by"| HARNESS
    HARNESS <-->|"step / observe"| ALE
    RUNNER <-->|"commit / reset"| GIT
    RUNNER -->|"append"| LOG
    LOG -->|"history in prompt"| RUNNER

    style AUTORESEARCH fill:#1a1a2e,stroke:#533483,color:#fff
    style LLM fill:#9C27B0,stroke:#fff,color:#fff
    style ALE fill:#1B5E20,stroke:#fff,color:#fff
    style GIT fill:#263238,stroke:#fff,color:#fff
    style LOG fill:#263238,stroke:#fff,color:#fff
    style USER fill:#E65100,stroke:#fff,color:#fff
```

---

## 2. Project Anatomy

```mermaid
mindmap
  root((AutoResearch))
    **LLM Pretraining**
      prepare.py
        ClimbMix 400B data
        BPE tokenizer
        BPB evaluation metric
      train.py
        GPT transformer
        Muon + AdamW optimizer
        Flash Attention 3
        Sliding window SSSL
      program.md
        Research protocol
    **Atari Agent**
      prepare.py
        Environment factory
        30-episode evaluation
        Fixed seed = 42
      agent.py
        Ball detection x3 methods
        Trajectory prediction
        Velocity estimation
        Urgency-based paddle control
      run.py
        LLM orchestration loop
        Git commit and rollback
        Rate limit handling
        TSV results logging
        Video recording on keep
      program.md
        DQN constraints
    **Shared Philosophy**
      One modifiable file
      Fixed evaluation harness
      Keep-or-discard protocol
      Git as experiment journal
```

---

## 3. The Experiment Loop

The core algorithm. Every autonomous experiment follows this exact path.

```mermaid
flowchart TD
    START(["Start"]) --> INIT["Initialize branch & results log"]
    INIT --> BASELINE["Run baseline evaluation"]

    BASELINE -->|"crashed"| ABORT(["Fix agent.py manually"])
    BASELINE -->|"success"| SET_BEST["best_reward = baseline score"]
    SET_BEST --> VID_BASE["Record baseline video"]

    VID_BASE --> LIMIT{Max experiments<br/>reached?}
    LIMIT -->|"yes"| DONE(["Print Summary"])
    LIMIT -->|"no"| READ["Read results.tsv history"]

    READ --> PROMPT["Build prompt:<br/>best code + history + reward"]
    PROMPT --> LLM["Call LLM API"]

    LLM -->|"retries exhausted"| SKIP["Skip this experiment"]
    LLM -->|"response received"| EXTRACT["Extract code from<br/>markdown response"]

    EXTRACT -->|"no code block"| SKIP
    EXTRACT -->|"valid Python"| WRITE["Overwrite agent.py"]

    WRITE --> GIT_COMMIT["git commit -m 'try: ...'"]
    GIT_COMMIT --> EVAL["Run evaluation<br/>30 episodes, 10min timeout"]

    EVAL -->|"crashed / timeout"| CRASH["Log crash<br/>git reset --hard"]
    EVAL -->|"reward > best"| KEEP["Log keep<br/>Update best"]
    EVAL -->|"reward <= best"| DISCARD["Log discard<br/>git reset --hard"]

    KEEP --> VID["Record video"]
    VID --> LIMIT

    CRASH --> LIMIT
    DISCARD --> LIMIT
    SKIP --> LIMIT

    style START fill:#4CAF50,color:#fff
    style DONE fill:#2196F3,color:#fff
    style ABORT fill:#f44336,color:#fff
    style KEEP fill:#4CAF50,color:#fff
    style VID fill:#2196F3,color:#fff
    style DISCARD fill:#FF9800,color:#fff
    style CRASH fill:#f44336,color:#fff
    style LLM fill:#9C27B0,color:#fff
```

---

## 4. Single Experiment Lifecycle

Every interaction between components during one experiment, from prompt to decision.

```mermaid
sequenceDiagram
    autonumber
    participant R as run.py
    participant FS as File System
    participant LLM as Claude API
    participant G as Git
    participant A as agent.py
    participant E as ALE Environment

    R->>FS: Read results.tsv + agent.py
    FS-->>R: History + best code

    R->>LLM: System prompt + code + history
    Note over LLM: temperature=0.7<br/>max_tokens=4096

    alt Rate Limited
        LLM-->>R: 429 (resets at: timestamp)
        R->>R: Sleep until reset
        R->>LLM: Retry
    end

    LLM-->>R: Complete new agent.py

    R->>R: Extract Python from markdown
    R->>FS: Write new agent.py
    R->>G: git add + commit

    R->>A: subprocess: python agent.py
    Note over A: train() runs up to 3600s (1 hour)

    loop 30 Episodes (seed 42..71)
        A->>E: env.reset(seed)
        A->>A: agent.reset()
        loop Until terminated
            A->>A: action = agent.act(obs)
            A->>E: env.step(action)
            E-->>A: obs, reward, done
        end
    end

    A-->>R: stdout: mean_reward=X.XXXX

    alt reward > best_reward
        R->>FS: Log "keep"
        R->>E: Record video (1 episode)
        Note over R: New best!
    else reward <= best_reward
        R->>FS: Log "discard"
        R->>G: git reset --hard
    else crashed
        R->>FS: Log "crash"
        R->>G: git reset --hard
    end
```

---

## 5. Experiment State Machine

Every possible state an experiment can be in, and every transition between them.

```mermaid
stateDiagram-v2
    [*] --> Pending: New experiment

    Pending --> PromptBuilt: Build prompt with history

    PromptBuilt --> CallingLLM: Send to API

    CallingLLM --> CallingLLM: Rate limited / retry
    CallingLLM --> Skipped: All retries exhausted
    CallingLLM --> CodeReceived: Response OK

    CodeReceived --> Skipped: No code block found
    CodeReceived --> CodeExtracted: Valid Python extracted

    CodeExtracted --> Committed: Write file + git commit

    Committed --> Evaluating: Launch subprocess

    Evaluating --> Crashed: Non-zero exit / timeout
    Evaluating --> Evaluated: mean_reward parsed

    Crashed --> Reverted: git reset --hard
    Evaluated --> Kept: reward > best
    Evaluated --> Discarded: reward <= best
    Discarded --> Reverted: git reset --hard

    Kept --> RecordingVideo: Record gameplay
    RecordingVideo --> [*]: Continue (new baseline)
    Reverted --> [*]: Continue (old baseline)
    Skipped --> [*]: Continue

    state Kept {
        [*] --> UpdateBestReward
        UpdateBestReward --> UpdateBestCode
        UpdateBestCode --> LogKeep
    }

    state Crashed {
        [*] --> LogCrash
        LogCrash --> SaveErrorTail
    }
```

---

## 6. Developer Journey

What it feels like to use AutoResearch, from setup to results.

```mermaid
journey
    title AutoResearch: From Setup to Discovery
    section Setup
      Clone the repo: 5: Researcher
      Install deps (uv sync): 4: Researcher
      Verify env (python prepare.py): 5: Researcher
      Set API key: 3: Researcher
    section Launch
      Run the autonomous loop: 5: Researcher
      Watch baseline evaluation: 4: Researcher, System
      First LLM call goes out: 5: System
    section Experimentation
      LLM proposes deeper CNN: 4: LLM
      Evaluation runs 30 episodes: 3: System
      Result improves - KEPT: 5: System
      Video recorded automatically: 5: System
      LLM proposes bad LR: 2: LLM
      Agent crashes - ROLLED BACK: 1: System
      LLM learns from history: 4: LLM
      New approach works: 5: LLM, System
    section Review
      Check results.tsv: 5: Researcher
      Watch videos of each improvement: 5: Researcher
      Browse git log of kept commits: 5: Researcher
      Analyze winning strategies: 5: Researcher
```

---

## 7. Agent Class Design

The full class hierarchy with all attributes, methods, and relationships.

```mermaid
classDiagram
    class Agent {
        +int noop = 0
        +int fire = 1
        +int right = 2
        +int left = 3
        -int _steps
        -bool _fired
        -list _ball_history
        -float _paddle_x
        -int _ball_lost_count
        -tuple _ball_velocity
        -list _velocity_samples
        -int _last_fire_time
        -tuple _stable_ball_pos
        -int _detection_confidence
        +__init__(action_space)
        +reset()
        +act(obs) int
        -_find_ball_enhanced(obs) tuple
        -_update_ball_tracking(x, y)
        -_calculate_target_position(x, y) float
        -_move_paddle_smooth(target, ball_y) int
        -_find_paddle_x(obs) float
    }

    class EvalHarness {
        <<frozen>>
        +str GAME
        +int NUM_EVAL_EPISODES = 30
        +int MAX_STEPS_PER_EPISODE = 10000
        +int TIME_BUDGET = 3600
        +int SEED = 42
        +make_env() Env
        +evaluate_agent(agent) dict
    }

    class AutonomousRunner {
        +str SYSTEM_PROMPT
        +main()
        +build_prompt() str
        +call_llm_with_retry() str
        +extract_code(response) str
        +run_evaluation() tuple
        +record_video() void
        +log_result()
        -git_commit() str
        -git_reset_hard()
    }

    class LiteLLM {
        <<external>>
        +completion() Response
    }

    class GitRepo {
        <<external>>
        +add()
        +commit()
        +reset()
    }

    AutonomousRunner --> Agent : modifies code
    EvalHarness --> Agent : evaluates
    AutonomousRunner --> EvalHarness : invokes
    AutonomousRunner --> LiteLLM : calls
    AutonomousRunner --> GitRepo : manages
```

---

## 8. Observation Pipeline

How a raw 210x160 RGB frame becomes a paddle movement action.

```mermaid
flowchart LR
    subgraph IN["Input"]
        OBS["RGB Frame<br/>210 x 160 x 3<br/>uint8"]
    end

    subgraph DETECT["Detection"]
        direction TB
        BALL["Ball Detection<br/><i>3 color methods</i>"]
        PAD["Paddle Detection<br/><i>brightness scan</i>"]
    end

    subgraph STATE["State Update"]
        direction TB
        TRACK["Position History<br/><i>last 5 frames</i>"]
        VEL["Velocity Estimate<br/><i>median filter</i>"]
        CONF["Detection<br/>Confidence"]
    end

    subgraph DECIDE["Decision"]
        direction TB
        TARGET["Target Position<br/><i>prediction + bounce</i>"]
        MOVE["Movement<br/><i>urgency threshold</i>"]
    end

    subgraph OUT["Output"]
        ACT{{"Action<br/>0|1|2|3"}}
    end

    OBS --> BALL & PAD
    BALL -->|"(x,y)"| TRACK --> VEL
    BALL -->|"confidence"| CONF
    VEL & CONF -->|"velocity + stable_pos"| TARGET
    PAD -->|"paddle_x"| MOVE
    TARGET -->|"target_x"| MOVE
    MOVE --> ACT

    style IN fill:#263238,color:#fff
    style DETECT fill:#1B5E20,color:#fff
    style STATE fill:#E65100,color:#fff
    style DECIDE fill:#4A148C,color:#fff
    style OUT fill:#B71C1C,color:#fff
```

---

## 9. Ball Detection Algorithm

The multi-method detection in `_find_ball_enhanced()` — three parallel strategies merged into one result.

```mermaid
flowchart TD
    FRAME["Full Frame 210x160x3"] --> CROP["Crop to game field<br/>rows 93:187, cols 6:154"]

    CROP --> M1["White Pixels<br/>R,G,B >= 200"]
    CROP --> M2["High Brightness<br/>max(R,G,B) >= 190"]
    CROP --> M3["Orange Ball<br/>R>200, G:100-180, B<100"]

    M1 & M2 & M3 --> OR["Combine: white OR bright OR orange"]

    OR --> COORDS["np.argwhere → pixel coordinates"]

    COORDS --> CHECK1{count == 0<br/>or > 30?}
    CHECK1 -->|"yes"| FAIL1["Return None"]

    CHECK1 -->|"no"| CHECK2{count > 4?}
    CHECK2 -->|"no"| CENTER["Mean of coordinates"]
    CHECK2 -->|"yes"| OUTLIER["Remove outliers<br/>median + 80th pctl distance"]

    OUTLIER --> CHECK3{any left?}
    CHECK3 -->|"no"| FAIL2["Return None"]
    CHECK3 -->|"yes"| CENTER

    CENTER --> OFFSET["Add crop offset<br/>y+93, x+6"]
    OFFSET --> BOUNDS{In game<br/>bounds?}
    BOUNDS -->|"no"| FAIL3["Return None"]
    BOUNDS -->|"yes"| OK["Return (x, y)"]

    style FAIL1 fill:#f44336,color:#fff
    style FAIL2 fill:#f44336,color:#fff
    style FAIL3 fill:#f44336,color:#fff
    style OK fill:#4CAF50,color:#fff
    style M1 fill:#E3F2FD,color:#000
    style M2 fill:#FFF3E0,color:#000
    style M3 fill:#FBE9E7,color:#000
```

---

## 10. Trajectory Prediction

How `_calculate_target_position()` decides where the paddle should be.

```mermaid
flowchart TD
    START["ball_x, ball_y"] --> FAR{ball_y < 125?}

    FAR -->|"yes — ball far away"| SIMPLE["target = ball_x<br/><i>simple tracking</i>"]

    FAR -->|"no — ball in lower half"| HAS_VEL{velocity<br/>available?}

    HAS_VEL -->|"no"| HAS_HIST{history >= 2?}
    HAS_HIST -->|"no"| DIRECT["target = ball_x"]
    HAS_HIST -->|"yes"| ANTIC["target = ball_x + delta * 0.2-0.4"]

    HAS_VEL -->|"yes"| APPROACHING{vy > 0.1?}
    APPROACHING -->|"no"| HAS_HIST

    APPROACHING -->|"yes"| CALC["time = (189 - y) / vy<br/>predicted_x = x + vx * time"]
    CALC --> BOUNCE["Reflect off walls<br/>x=6 and x=154"]
    BOUNCE --> NEAR{ball_y > 165<br/>and |vx| > 0.5?}
    NEAR -->|"yes"| OFFSET["Strategic offset<br/>for angle control"]
    NEAR -->|"no"| CLAMP
    OFFSET --> CLAMP["Clamp to [10, 150]"]

    SIMPLE & DIRECT & ANTIC & CLAMP --> SMOOTH

    subgraph SMOOTH["_move_paddle_smooth()"]
        direction LR
        DIST["distance = target - paddle_x"]
        DIST --> URG{ball urgency}
        URG -->|"y>170"| T1["1px threshold"]
        URG -->|"y>150"| T2["2px threshold"]
        URG -->|"y>130"| T3["3.5px threshold"]
        URG -->|"y<=130"| T4["5px threshold"]
        T1 & T2 & T3 & T4 --> MOVE{|dist| > threshold?}
        MOVE -->|"no"| NOOP["NOOP"]
        MOVE -->|"yes, dist>0"| RIGHT["RIGHT"]
        MOVE -->|"yes, dist<0"| LEFT["LEFT"]
    end

    style NOOP fill:#607D8B,color:#fff
    style RIGHT fill:#2196F3,color:#fff
    style LEFT fill:#FF9800,color:#fff
```

---

## 11. Evaluation Harness

The immutable `evaluate_agent()` protocol — the ground truth metric.

```mermaid
flowchart TD
    CALL(["evaluate_agent(agent)"]) --> ENV["make_env()<br/>Breakout-v5, frameskip=1<br/>sticky_actions=0.25"]

    ENV --> INIT["rewards=[], steps=[]"]
    INIT --> LOOP{ep < 30?}

    LOOP -->|"no"| CLOSE["env.close()"]
    LOOP -->|"yes"| RESET["env.reset(seed=42+ep)<br/>agent.reset()"]

    RESET --> STEP{done or<br/>steps >= 10k?}
    STEP -->|"yes"| SAVE["Save episode reward & steps"]
    SAVE --> INC["ep += 1"] --> LOOP

    STEP -->|"no"| ACT["action = agent.act(obs)"]
    ACT --> ENVSTEP["obs, reward, done = env.step(action)"]
    ENVSTEP --> ACCUM["total_reward += reward"] --> STEP

    CLOSE --> STATS

    subgraph STATS["Returns"]
        S1["mean_reward"]
        S2["std_reward"]
        S3["min/max_reward"]
        S4["mean_steps"]
        S5["total_steps"]
    end

    style CALL fill:#4CAF50,color:#fff
    style STATS fill:#E8F5E9,color:#000
```

---

## 12. LLM Prompt Protocol

What gets sent to the LLM, and how the response is processed.

```mermaid
flowchart LR
    subgraph SYS["System Prompt (Fixed)"]
        direction TB
        S1["Role: AI researcher"]
        S2["Goal: maximize mean_reward"]
        S3["Constraint: must use DQN"]
        S4["Available imports"]
        S5["Experiment suggestions:<br/>architecture, double DQN,<br/>prioritized replay, etc."]
        S6["Format: complete agent.py<br/>in python code block"]
    end

    subgraph USR["User Prompt (Dynamic)"]
        direction TB
        U1["Current best agent.py"]
        U2["Current best mean_reward"]
        U3["Full results.tsv"]
        U4["Experiment number"]
    end

    subgraph RESP["Response Processing"]
        direction TB
        R1["Regex: python code block"]
        R2["Fallback: generic code block"]
        R3["Extracted agent.py"]
    end

    SYS --> API["litellm.completion()<br/>temp=0.7, max=4096"]
    USR --> API
    API --> RESP

    style SYS fill:#1A237E,color:#fff
    style USR fill:#004D40,color:#fff
    style RESP fill:#311B92,color:#fff
    style API fill:#9C27B0,color:#fff
```

---

## 13. Git Branching Strategy

The main branch is a clean record of validated improvements. Failed experiments are erased.

```mermaid
gitGraph
    commit id: "baseline" tag: "v0"
    commit id: "try: deeper CNN" type: HIGHLIGHT
    commit id: "try: double DQN"
    branch "discarded-1"
    commit id: "DISCARDED" type: REVERSE
    checkout main
    commit id: "try: prioritized replay" type: HIGHLIGHT
    commit id: "try: noisy nets"
    branch "crashed-1"
    commit id: "CRASHED" type: REVERSE
    checkout main
    commit id: "try: reward shaping" type: HIGHLIGHT
    commit id: "try: lr=0.0001"
    branch "discarded-2"
    commit id: "DISCARDED 2" type: REVERSE
    checkout main
    commit id: "try: dueling DQN" type: HIGHLIGHT
```

> **Key insight:** The main branch only contains improvements. `git log` reads like a scientific paper — each commit is a validated finding. Failed experiments exist only in `results.tsv`.

---

## 14. Experiment Timeline

How time is allocated during a single experiment cycle.

```mermaid
gantt
    title Single Experiment Timeline
    dateFormat X
    axisFormat %s

    section LLM
    Build prompt             :a1, 0, 5
    API call + response      :a2, 5, 30
    Extract code             :a3, 30, 32

    section Git
    Write + commit           :b1, 32, 34

    section Training
    agent train() up to 1hr  :crit, c1, 34, 3634

    section Evaluation
    Episodes 1-10            :d1, 3634, 3734
    Episodes 11-20           :d2, 3734, 3834
    Episodes 21-30           :d3, 3834, 3934

    section Video
    Record if kept           :v1, 3934, 3964

    section Decision
    Compare + log            :e1, 3964, 3968
    Keep or reset            :e2, 3968, 3970
```

---

## 15. Data Model

The entities and relationships in the system.

```mermaid
erDiagram
    EXPERIMENT ||--o| RESULT : produces
    EXPERIMENT {
        int number PK
        string agent_code
        string description
        string llm_response
    }
    RESULT {
        string commit FK
        float mean_reward
        string status "keep|discard|crash"
        string description
    }
    AGENT ||--|| EXPERIMENT : modified_by
    AGENT {
        string file_path "atari/agent.py"
        string current_code
        float best_reward
    }
    EVALUATION ||--|| RESULT : determines
    EVALUATION {
        int episodes "30"
        int seed "42"
        float mean_reward
        float std_reward
        float min_reward
        float max_reward
        int total_steps
    }
    GIT_COMMIT ||--o| RESULT : tracks
    GIT_COMMIT {
        string hash PK
        string message "try: ..."
        string branch
        boolean survived
    }
    LLM_CALL ||--|| EXPERIMENT : generates
    LLM_CALL {
        string model
        float temperature "0.7"
        int max_tokens "4096"
        string system_prompt
        string user_prompt
        int retry_count
    }
    VIDEO ||--o| RESULT : records
    VIDEO {
        string filename "expNNNN_rewardX.mp4"
        int experiment_num
        float reward
    }
```

---

## 16. Action Space

Breakout's four actions and their approximate usage distribution during play.

```mermaid
pie title Approximate Action Distribution
    "NOOP (0) — Aligned" : 35
    "FIRE (1) — Launch ball" : 5
    "RIGHT (2) — Chase right" : 30
    "LEFT (3) — Chase left" : 30
```

```mermaid
flowchart LR
    subgraph ACTIONS["Breakout Actions"]
        A0["**NOOP (0)**<br/>Paddle is already<br/>at target position"]
        A1["**FIRE (1)**<br/>Launch ball at start<br/>or after losing life"]
        A2["**RIGHT (2)**<br/>Target is to the<br/>right of paddle"]
        A3["**LEFT (3)**<br/>Target is to the<br/>left of paddle"]
    end

    style A0 fill:#607D8B,color:#fff
    style A1 fill:#f44336,color:#fff
    style A2 fill:#2196F3,color:#fff
    style A3 fill:#FF9800,color:#fff
```

---

## 17. Error Handling & Recovery

Every failure mode the system can encounter, mapped to its recovery strategy.

```mermaid
flowchart LR
    subgraph FAIL["Failure Modes"]
        F1["Rate Limited<br/><i>429 response</i>"]
        F2["API Error<br/><i>network, auth, etc.</i>"]
        F3["Bad Response<br/><i>no code block</i>"]
        F4["Agent Crash<br/><i>runtime error</i>"]
        F5["Timeout<br/><i>> 75 minutes</i>"]
        F6["No Improvement<br/><i>reward <= best</i>"]
    end

    subgraph RECOVER["Recovery"]
        R1["Parse reset timestamp<br/>Sleep until window opens"]
        R2["Exponential backoff<br/>30s, 60s, ... 300s<br/>Up to 10 retries"]
        R3["Skip experiment"]
        R4["Log crash<br/>git reset --hard"]
        R5["Treat as crash"]
        R6["Log discard<br/>git reset --hard"]
    end

    F1 --> R1
    F2 --> R2
    F3 --> R3
    F4 --> R4
    F5 --> R5
    F6 --> R6

    R1 & R2 & R3 & R4 & R5 & R6 --> NEXT["Next Experiment"]

    style FAIL fill:#FFEBEE,color:#000
    style RECOVER fill:#E8F5E9,color:#000
    style NEXT fill:#4CAF50,color:#fff
```

---

## 18. Dual-Track Comparison

The platform supports two research tracks with the same philosophy but different domains.

```mermaid
flowchart TB
    subgraph LLM["Track 1: LLM Pretraining"]
        direction TB
        L1["Modifiable: **train.py**"]
        L2["Metric: **val_bpb** (lower = better)"]
        L3["Budget: 5 min training (LLM track)"]
        L4["Architecture: GPT Transformer"]
        L5["Eval: Validation loss on<br/>held-out ClimbMix shard"]
        L6["Optimizer: Muon + AdamW"]
    end

    subgraph ATARI["Track 2: Atari Game Playing"]
        direction TB
        A1["Modifiable: **agent.py**"]
        A2["Metric: **mean_reward** (higher = better)"]
        A3["Budget: 1 hour train + eval"]
        A4["Architecture: Heuristic / DQN"]
        A5["Eval: 30 episodes<br/>ALE/Breakout-v5, seed 42"]
        A6["Runner: run.py (autonomous)"]
    end

    subgraph SHARED["Shared Principles"]
        S1["Single modifiable file"]
        S2["Fixed evaluation harness"]
        S3["LLM-driven optimization"]
        S4["Git version control"]
        S5["Keep-or-discard protocol"]
        S6["TSV provenance log"]
    end

    LLM ~~~ SHARED ~~~ ATARI

    style LLM fill:#0D47A1,color:#fff
    style ATARI fill:#4A148C,color:#fff
    style SHARED fill:#1B5E20,color:#fff
```

---

## 19. Dependency Graph

How all files and external packages connect.

```mermaid
flowchart BT
    subgraph EXT["External"]
        GYM["gymnasium + ale-py"]
        TORCH["PyTorch"]
        NP["NumPy"]
        LIT["litellm"]
        CLAUDE["Claude API"]
    end

    subgraph FILES["atari/"]
        PREP["prepare.py"]
        AGENT["agent.py"]
        RUN["run.py"]
        RES["results.tsv"]
        LOG["run.log"]
        VID["videos/"]
    end

    GYM & NP --> PREP
    NP & TORCH --> AGENT
    PREP --> AGENT
    AGENT --> RUN
    LIT --> RUN
    CLAUDE --> LIT
    RUN --> RES & LOG & VID

    style EXT fill:#37474F,color:#fff
    style FILES fill:#1A237E,color:#fff
```

---

## 20. Outcome Space

A quadrant view of where experiments land based on execution success and reward.

```mermaid
quadrantChart
    title Experiment Outcome Space
    x-axis "Lower Reward" --> "Higher Reward"
    y-axis "Crashed" --> "Ran Successfully"
    quadrant-1 "KEEP"
    quadrant-2 "DISCARD"
    quadrant-3 "CRASH"
    quadrant-4 "CRASH"
    "Deeper CNN": [0.75, 0.9]
    "Double DQN": [0.85, 0.95]
    "Bad LR": [0.3, 0.85]
    "OOM": [0.1, 0.15]
    "Timeout": [0.2, 0.1]
    "Baseline": [0.5, 0.9]
    "Noisy nets": [0.45, 0.88]
    "Reward shaping": [0.7, 0.92]
    "Syntax error": [0.05, 0.05]
```

---

## Design Philosophy

This system encodes the scientific method into an automated loop:

| Principle | Implementation |
|-----------|----------------|
| **Hypothesis** | LLM proposes a code modification |
| **Experiment** | Fixed 30-episode evaluation with seeded environment |
| **Observation** | mean_reward compared to current best |
| **Conclusion** | Keep (publish) or discard (retract) |
| **Reproducibility** | Deterministic seeds, git history, TSV log |
| **Iteration** | Full history fed back into next hypothesis |

The result: **git log reads like a research paper** — each surviving commit is a validated improvement, and `results.tsv` is the lab notebook with every attempt documented.
