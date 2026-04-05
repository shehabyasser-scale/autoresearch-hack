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
    # Adaptive paddle positioning with momentum-based movement and trajectory confidence
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
        self._paddle_momentum = 0
        self._last_paddle_x = 80.0
        self._confidence_score = 0

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.

        Enhanced strategy: adaptive positioning with momentum and confidence-based movement.
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
            self._confidence_score = max(0, self._confidence_score - 0.3)
            
            # Use predicted position if we have trajectory data
            if len(self._ball_history) >= 2 and self._predicted_ball_x is not None and self._confidence_score > 0.2:
                ball_x = self._predicted_ball_x
                # Decay prediction accuracy over time
                if self._no_ball_count > 3:
                    ball_x = None
            
            if ball_x is None:
                if self._no_ball_count > 12:
                    self._no_ball_count = 0
                    self._paddle_momentum = 0
                    return self.fire
                # Continue momentum-based movement
                if abs(self._paddle_momentum) > 0.5:
                    self._paddle_momentum *= 0.8  # Decay momentum
                    return self.right if self._paddle_momentum > 0 else self.left
                return self.noop
        else:
            self._no_ball_count = 0
            self._confidence_score = min(1.0, self._confidence_score + 0.4)

        # Update ball history and compute smoothed velocity
        self._ball_history.append(ball_x)
        if len(self._ball_history) > 6:
            self._ball_history.pop(0)

        # Compute velocity using weighted average of recent measurements
        velocity = 0
        if len(self._ball_history) >= 3:
            # Use weighted linear regression for more stable velocity
            positions = np.array(self._ball_history[-4:])
            times = np.arange(len(positions))
            weights = np.linspace(0.5, 1.0, len(positions))  # More weight to recent positions
            if len(positions) > 1:
                # Weighted least squares
                A = np.vstack([times, np.ones(len(times))]).T
                W = np.diag(weights)
                velocity = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ positions, rcond=None)[0][0]
        
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 4:
            self._velocity_history.pop(0)
        
        # Smooth velocity with confidence weighting
        if self._velocity_history:
            recent_velocities = np.array(self._velocity_history)
            weights = np.linspace(0.6, 1.0, len(recent_velocities))
            avg_velocity = np.average(recent_velocities, weights=weights)
        else:
            avg_velocity = 0

        # Find paddle position
        paddle_x = self._find_paddle_x(obs)
        self._last_paddle_x = paddle_x

        # Dynamic lookahead based on velocity and confidence
        base_lookahead = 2
        velocity_factor = min(2.0, abs(avg_velocity) * 0.5)
        confidence_factor = self._confidence_score
        lookahead = base_lookahead + velocity_factor * confidence_factor
        
        self._predicted_ball_x = ball_x + avg_velocity * lookahead
        
        # Adaptive threshold based on ball speed and confidence
        base_threshold = 1.5
        speed_factor = min(1.5, abs(avg_velocity) * 0.3)
        threshold = base_threshold + speed_factor * confidence_factor
        
        # Calculate desired movement
        target_offset = self._predicted_ball_x - paddle_x
        
        # Momentum-based movement decision
        if abs(target_offset) > threshold:
            desired_direction = 1 if target_offset > 0 else -1
            
            # Update momentum
            self._paddle_momentum = 0.7 * self._paddle_momentum + 0.3 * desired_direction
            
            # Make movement decision based on momentum
            if self._paddle_momentum > 0.3:
                return self.right
            elif self._paddle_momentum < -0.3:
                return self.left
            else:
                return self.noop
        else:
            # Close to target, reduce momentum
            self._paddle_momentum *= 0.5
            
            # Fine positioning for close targets
            if abs(target_offset) > 0.5:
                return self.right if target_offset > 0 else self.left
            else:
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
            return self._last_paddle_x
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