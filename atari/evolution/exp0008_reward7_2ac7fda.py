"""
Atari agent — the file the AI researcher modifies.

This is the equivalent of train.py in the LLM autoresearch setup.
Modify anything here: the agent's strategy, observation preprocessing,
internal state, heuristics, learned components, etc.

Usage: python agent.py
"""

import time
import numpy as np
from prepare import GAME, TIME_BUDGET, make_env, evaluate_agent

# ---------------------------------------------------------------------------
# Agent (MODIFY THIS)
# ---------------------------------------------------------------------------

class Agent:
    # Improved ball tracking with better paddle movement and ball prediction
    """
    Atari Breakout agent.

    The AI researcher should modify this class to maximize mean_reward
    on the fixed evaluation harness (30 episodes of Breakout-v5).

    Interface:
        agent.act(obs) -> action (int)
        agent.reset()  -> None (called at start of each episode)
    """

    def __init__(self, action_space):
        self.action_space = action_space
        # Breakout actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        self.noop = 0
        self.fire = 1
        self.right = 2
        self.left = 3

    def reset(self):
        """Called at the start of each episode."""
        self._steps = 0
        self._fired = False
        self._last_ball_x = None
        self._ball_velocity = 0
        self._no_ball_count = 0

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.

        Improved strategy: better ball tracking with velocity estimation
        and more aggressive paddle movement.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Find ball position (look between bricks and paddle)
        ball_x = self._find_ball_x(obs)
        
        if ball_x is None:
            self._no_ball_count += 1
            # Only fire if we haven't seen the ball for several frames
            if self._no_ball_count > 10:
                self._no_ball_count = 0
                return self.fire
            # Otherwise, keep last known movement
            if hasattr(self, '_last_action') and self._last_action in [self.left, self.right]:
                return self._last_action
            return self.noop
        else:
            self._no_ball_count = 0

        # Estimate ball velocity
        if self._last_ball_x is not None:
            self._ball_velocity = ball_x - self._last_ball_x
        self._last_ball_x = ball_x

        # Find paddle position
        paddle_x = self._find_paddle_x(obs)

        # Predict where ball will be and move paddle there
        predicted_ball_x = ball_x + self._ball_velocity * 2
        
        # More aggressive movement - move even for small differences
        if predicted_ball_x < paddle_x - 1:
            self._last_action = self.left
            return self.left
        elif predicted_ball_x > paddle_x + 1:
            self._last_action = self.right
            return self.right
        else:
            self._last_action = self.noop
            return self.noop

    def _find_ball_x(self, obs):
        """Find ball x-coordinate in the playing field, or None if not visible."""
        # Ball region: between bricks (row ~100) and paddle (row ~185)
        field = obs[100:185, 8:152, :]
        bright = np.max(field, axis=2) > 180
        coords = np.argwhere(bright)
        
        # Filter out noise - ball should be small bright object
        if len(coords) == 0 or len(coords) > 30:
            return None
            
        # Look for compact bright regions (likely the ball)
        if len(coords) <= 8:  # Ball is typically 2-4 pixels
            return float(np.mean(coords[:, 1])) + 8
            
        # If too many bright pixels, try to find the most compact cluster
        x_coords = coords[:, 1]
        x_center = np.mean(x_coords)
        distances = np.abs(x_coords - x_center)
        close_coords = coords[distances <= 3]  # Within 3 pixels of center
        
        if len(close_coords) > 0:
            return float(np.mean(close_coords[:, 1])) + 8
            
        return None

    def _find_paddle_x(self, obs):
        """Find paddle x-center."""
        paddle_strip = obs[189:194, :, :]
        bright = np.max(paddle_strip, axis=2) > 180
        coords = np.argwhere(bright)
        if len(coords) == 0:
            return 80.0
        return float(np.mean(coords[:, 1]))


# ---------------------------------------------------------------------------
# Train function (optional — for strategies that need training time)
# ---------------------------------------------------------------------------

def train():
    """
    Optional training phase, runs within TIME_BUDGET seconds.

    Use this if your agent needs to learn (e.g., evolve weights,
    tune parameters via trial runs, etc.). The evaluation happens
    AFTER this function returns.

    For pure heuristic agents, this can be a no-op.
    """
    print(f"Training budget: {TIME_BUDGET}s")
    print("Current agent is heuristic-only, no training needed.")
    print("(Modify this function to add learning: evolutionary search, "
          "policy gradients, parameter tuning, etc.)")


# ---------------------------------------------------------------------------
# Main — run training + evaluation (equivalent to train.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Phase 1: Training (within time budget)
    print("=" * 50)
    print("PHASE 1: Training")
    print("=" * 50)
    train()
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")
    print()

    # Phase 2: Fixed evaluation
    print("=" * 50)
    print("PHASE 2: Evaluation")
    print("=" * 50)
    env = make_env()
    agent = Agent(env.action_space)
    env.close()

    results = evaluate_agent(agent)
    eval_time = time.time() - t0 - train_time
    total_time = time.time() - t0

    # Print results in parseable format (matches autoresearch convention)
    print()
    print("---")
    print(f"mean_reward:      {results['mean_reward']:.4f}")
    print(f"std_reward:       {results['std_reward']:.4f}")
    print(f"min_reward:       {results['min_reward']:.4f}")
    print(f"max_reward:       {results['max_reward']:.4f}")
    print(f"mean_steps:       {results['mean_steps']:.1f}")
    print(f"total_steps:      {results['total_steps']}")
    print(f"episodes:         {results['episodes']}")
    print(f"training_seconds: {train_time:.1f}")
    print(f"eval_seconds:     {eval_time:.1f}")
    print(f"total_seconds:    {total_time:.1f}")