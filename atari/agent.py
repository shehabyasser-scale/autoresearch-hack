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
    # Simplified robust tracking with improved trajectory prediction and adaptive positioning
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
        self._ball_positions = []
        self._no_ball_count = 0
        self._last_paddle_x = 80.0
        self._trajectory_confidence = 0.0
        self._last_action = self.noop
        self._action_momentum = 0
        self._ball_lost_frames = 0

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.
        
        Simplified robust strategy focusing on reliable ball tracking and smart positioning.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Find ball and paddle positions
        ball_pos = self._find_ball_robust(obs)
        paddle_x = self._find_paddle_x(obs)
        self._last_paddle_x = paddle_x

        if ball_pos is None:
            self._no_ball_count += 1
            self._ball_lost_frames += 1
            self._trajectory_confidence = max(0, self._trajectory_confidence - 0.4)
            
            # Emergency ball recovery
            if self._ball_lost_frames > 15:
                return self.fire
            
            # Use last known trajectory if confident
            if len(self._ball_positions) >= 3 and self._trajectory_confidence > 0.3:
                predicted_x = self._predict_intercept()
                if predicted_x is not None:
                    return self._move_towards_target(paddle_x, predicted_x, urgent=True)
            
            # Default to center when completely lost
            return self._move_towards_target(paddle_x, 80, urgent=False)
        else:
            ball_x, ball_y = ball_pos
            self._no_ball_count = 0
            self._ball_lost_frames = 0
            self._trajectory_confidence = min(1.0, self._trajectory_confidence + 0.5)

        # Update ball position history
        self._ball_positions.append((ball_x, ball_y, self._steps))
        if len(self._ball_positions) > 8:
            self._ball_positions.pop(0)

        # Calculate target position based on ball trajectory
        target_x = self._calculate_target_position(ball_x, ball_y)
        
        # Determine urgency based on ball Y position
        urgent = ball_y > 60  # Ball is getting close to paddle level
        
        return self._move_towards_target(paddle_x, target_x, urgent)

    def _find_ball_robust(self, obs):
        """Robust ball detection focusing on consistency."""
        # Focus on the main game area
        field = obs[95:180, 8:152, :]
        
        # Look for bright pixels (ball is typically white/bright)
        brightness = np.max(field, axis=2)
        bright_pixels = brightness > 180
        
        # Also check for high contrast pixels
        r, g, b = field[:, :, 0], field[:, :, 1], field[:, :, 2]
        white_like = (r > 170) & (g > 170) & (b > 170)
        
        # Combine criteria
        ball_candidates = bright_pixels | white_like
        coords = np.argwhere(ball_candidates)
        
        if len(coords) == 0:
            return None
            
        # Filter by size - ball should be small
        if len(coords) > 20:
            # Try stricter criteria
            very_bright = brightness > 200
            strict_coords = np.argwhere(very_bright)
            if len(strict_coords) > 0 and len(strict_coords) <= 15:
                coords = strict_coords
            else:
                return None
        
        if len(coords) <= 15:
            y_center = float(np.mean(coords[:, 0])) + 95  # Adjust for field offset
            x_center = float(np.mean(coords[:, 1])) + 8   # Adjust for field offset
            return (x_center, y_center)
            
        return None

    def _calculate_target_position(self, ball_x, ball_y):
        """Calculate where paddle should be positioned."""
        if len(self._ball_positions) < 3:
            return ball_x
            
        # Calculate velocity from recent positions
        recent_positions = self._ball_positions[-4:]
        if len(recent_positions) >= 2:
            # Use linear regression for more stable velocity estimation
            times = np.array([pos[2] for pos in recent_positions])
            x_positions = np.array([pos[0] for pos in recent_positions])
            
            if len(times) >= 3:
                # Fit line to get velocity
                dt = times - times[0]
                if dt[-1] > 0:
                    velocity_x = np.polyfit(dt, x_positions, 1)[0]
                else:
                    velocity_x = 0
            else:
                velocity_x = (x_positions[-1] - x_positions[0]) / max(1, times[-1] - times[0])
        else:
            velocity_x = 0

        # Predict intercept position
        predicted_x = self._predict_intercept_simple(ball_x, ball_y, velocity_x)
        
        return predicted_x

    def _predict_intercept_simple(self, ball_x, ball_y, velocity_x):
        """Simple physics-based intercept prediction."""
        paddle_y = 189  # Approximate paddle Y position
        
        if ball_y >= paddle_y:
            return ball_x
            
        # Estimate time to reach paddle
        frames_to_paddle = max(1, (paddle_y - ball_y) / 2.0)  # Assume ball moves ~2 pixels down per frame
        
        # Project ball position accounting for wall bounces
        predicted_x = ball_x + velocity_x * frames_to_paddle
        
        # Handle wall bounces
        game_width = 152 - 8  # Playable width
        left_wall = 8
        right_wall = 152
        
        # Simulate bounces
        while predicted_x < left_wall or predicted_x > right_wall:
            if predicted_x < left_wall:
                predicted_x = left_wall + (left_wall - predicted_x)
                velocity_x = -velocity_x * 0.9  # Slight energy loss
            elif predicted_x > right_wall:
                predicted_x = right_wall - (predicted_x - right_wall)
                velocity_x = -velocity_x * 0.9
        
        return predicted_x

    def _predict_intercept(self):
        """Predict intercept using stored trajectory data."""
        if len(self._ball_positions) < 3:
            return None
            
        # Get recent positions
        recent = self._ball_positions[-3:]
        x_positions = [pos[0] for pos in recent]
        y_positions = [pos[1] for pos in recent]
        
        # Calculate average velocity
        dx = (x_positions[-1] - x_positions[0]) / max(1, len(recent) - 1)
        dy = (y_positions[-1] - y_positions[0]) / max(1, len(recent) - 1)
        
        if abs(dy) < 0.1:  # Ball moving horizontally
            return x_positions[-1]
            
        # Predict intercept
        current_y = y_positions[-1]
        paddle_y = 189
        frames_to_intercept = (paddle_y - current_y) / max(0.1, dy)
        
        if frames_to_intercept > 0:
            predicted_x = x_positions[-1] + dx * frames_to_intercept
            # Simple wall bounce
            if predicted_x < 8:
                predicted_x = 16 - predicted_x
            elif predicted_x > 152:
                predicted_x = 304 - predicted_x
            return max(8, min(152, predicted_x))
            
        return x_positions[-1]

    def _move_towards_target(self, paddle_x, target_x, urgent=False):
        """Move paddle towards target with momentum and urgency consideration."""
        distance = target_x - paddle_x
        
        # Adjust threshold based on urgency and confidence
        base_threshold = 2.0
        if urgent:
            threshold = base_threshold * 0.6
        else:
            threshold = base_threshold * (1.2 - self._trajectory_confidence * 0.4)
        
        # Apply momentum for smoother movement
        if abs(distance) > threshold:
            desired_direction = 1 if distance > 0 else -1
            momentum_factor = 0.7 if urgent else 0.5
            self._action_momentum = momentum_factor * self._action_momentum + (1 - momentum_factor) * desired_direction
            
            move_threshold = 0.3 if urgent else 0.4
            
            if self._action_momentum > move_threshold:
                return self.right
            elif self._action_momentum < -move_threshold:
                return self.left
            else:
                return self.noop
        else:
            # Fine positioning
            self._action_momentum *= 0.6
            if abs(distance) > 1.0:
                return self.right if distance > 0 else self.left
            else:
                return self.noop

    def _find_paddle_x(self, obs):
        """Find paddle x-center."""
        paddle_region = obs[189:195, :, :]
        brightness = np.max(paddle_region, axis=2)
        bright_pixels = brightness > 150
        coords = np.argwhere(bright_pixels)
        
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