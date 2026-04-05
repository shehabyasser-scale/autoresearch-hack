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
    # Enhanced ball detection using multi-channel analysis and trajectory smoothing
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
        self._ball_history = []
        self._velocity_history = []
        self._no_ball_count = 0
        self._predicted_ball_x = None

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.

        Enhanced strategy: multi-channel ball detection with trajectory smoothing.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Find ball position using enhanced detection
        ball_x = self._find_ball_x_enhanced(obs)
        
        if ball_x is None:
            self._no_ball_count += 1
            # Use predicted position if we have trajectory data
            if len(self._ball_history) >= 2 and self._predicted_ball_x is not None:
                ball_x = self._predicted_ball_x
                # Decay confidence in prediction
                if self._no_ball_count > 5:
                    ball_x = None
            
            if ball_x is None:
                if self._no_ball_count > 15:
                    self._no_ball_count = 0
                    return self.fire
                # Keep moving in last direction if we had one
                if hasattr(self, '_last_action') and self._last_action in [self.left, self.right]:
                    return self._last_action
                return self.noop
        else:
            self._no_ball_count = 0

        # Update ball history and compute smoothed velocity
        self._ball_history.append(ball_x)
        if len(self._ball_history) > 5:
            self._ball_history.pop(0)

        # Compute velocity using multiple frames for stability
        velocity = 0
        if len(self._ball_history) >= 3:
            # Use linear regression on recent positions for smoother velocity
            positions = np.array(self._ball_history[-3:])
            times = np.arange(len(positions))
            if len(positions) > 1:
                velocity = np.polyfit(times, positions, 1)[0]
        
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 3:
            self._velocity_history.pop(0)
        
        # Smooth velocity estimate
        avg_velocity = np.mean(self._velocity_history) if self._velocity_history else 0

        # Find paddle position
        paddle_x = self._find_paddle_x(obs)

        # Predict ball position with longer lookahead
        lookahead = 3 if abs(avg_velocity) > 1 else 2
        self._predicted_ball_x = ball_x + avg_velocity * lookahead
        
        # More responsive movement with different thresholds based on velocity
        threshold = 2 if abs(avg_velocity) > 2 else 1
        
        if self._predicted_ball_x < paddle_x - threshold:
            self._last_action = self.left
            return self.left
        elif self._predicted_ball_x > paddle_x + threshold:
            self._last_action = self.right
            return self.right
        else:
            self._last_action = self.noop
            return self.noop

    def _find_ball_x_enhanced(self, obs):
        """Enhanced ball detection using multiple color channels and better filtering."""
        # Ball region: between bricks and paddle
        field = obs[95:185, 8:152, :]
        
        # Check multiple color channels - ball appears bright in different channels
        r_bright = field[:, :, 0] > 180
        g_bright = field[:, :, 1] > 180
        b_bright = field[:, :, 2] > 180
        
        # Ball is typically bright in at least one channel
        bright = r_bright | g_bright | b_bright
        
        # Also check for very bright pixels (ball center)
        very_bright = np.max(field, axis=2) > 200
        
        # Combine both criteria
        ball_candidates = bright | very_bright
        
        coords = np.argwhere(ball_candidates)
        
        if len(coords) == 0:
            return None
            
        # Filter by size - ball should be small
        if len(coords) > 25:
            # Too many pixels, try stricter criteria
            very_bright_coords = np.argwhere(very_bright)
            if len(very_bright_coords) > 0 and len(very_bright_coords) <= 10:
                coords = very_bright_coords
            else:
                return None
        
        # Find the most compact cluster
        if len(coords) <= 8:
            return float(np.mean(coords[:, 1])) + 8
            
        # For larger clusters, find the densest region
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        
        # Use 2D clustering to find ball
        center_x = np.median(x_coords)
        center_y = np.median(y_coords)
        
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        close_coords = coords[distances <= 4]
        
        if len(close_coords) >= 2:
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