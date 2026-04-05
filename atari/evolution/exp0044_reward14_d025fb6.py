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
    # Robust multi-method ball detection with simplified but more reliable positioning strategy
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
        self._last_action = self.noop
        self._ball_lost_frames = 0
        self._ball_velocity = (0, 0)
        self._last_ball_pos = None
        self._consistent_velocity = (0, 0)
        self._velocity_confidence = 0.0

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.
        
        Robust strategy with multi-method ball detection and reliable positioning.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Find ball and paddle positions using multiple methods
        ball_pos = self._find_ball_multi_method(obs)
        paddle_x = self._find_paddle_x(obs)
        self._last_paddle_x = paddle_x

        if ball_pos is None:
            self._no_ball_count += 1
            self._ball_lost_frames += 1
            self._velocity_confidence *= 0.8
            
            # Emergency ball recovery
            if self._ball_lost_frames > 15:
                return self.fire
            
            # Use last known trajectory if confident
            if self._velocity_confidence > 0.5 and self._last_ball_pos is not None:
                predicted_x = self._predict_from_last_known()
                if predicted_x is not None:
                    return self._move_towards_target(paddle_x, predicted_x, urgent=True)
            
            # Default positioning when ball is lost
            center_x = 80
            return self._move_towards_target(paddle_x, center_x, urgent=False)
        else:
            ball_x, ball_y = ball_pos
            self._no_ball_count = 0
            self._ball_lost_frames = 0

        # Update ball tracking
        self._update_ball_tracking(ball_x, ball_y)

        # Calculate target position
        target_x = self._calculate_target_position(ball_x, ball_y, paddle_x)
        
        # Determine urgency based on ball position and velocity
        urgent = ball_y > 70 or abs(self._consistent_velocity[1]) > 2.5
        
        return self._move_towards_target(paddle_x, target_x, urgent)

    def _find_ball_multi_method(self, obs):
        """Multi-method ball detection for robustness."""
        # Method 1: Standard bright pixel detection
        ball_pos = self._find_ball_method1(obs)
        if ball_pos is not None:
            return ball_pos
            
        # Method 2: Color-based detection
        ball_pos = self._find_ball_method2(obs)
        if ball_pos is not None:
            return ball_pos
            
        # Method 3: Edge-based detection
        ball_pos = self._find_ball_method3(obs)
        return ball_pos

    def _find_ball_method1(self, obs):
        """Standard bright pixel detection."""
        field = obs[95:180, 8:152, :]
        brightness = np.max(field, axis=2)
        bright_pixels = brightness > 180
        coords = np.argwhere(bright_pixels)
        
        if len(coords) == 0 or len(coords) > 20:
            return None
            
        y_center = float(np.mean(coords[:, 0])) + 95
        x_center = float(np.mean(coords[:, 1])) + 8
        return (x_center, y_center)

    def _find_ball_method2(self, obs):
        """Color-based detection looking for white/bright objects."""
        field = obs[95:180, 8:152, :]
        r, g, b = field[:, :, 0], field[:, :, 1], field[:, :, 2]
        
        # Look for white-ish pixels
        white_mask = (r > 160) & (g > 160) & (b > 160)
        # Also look for very bright pixels in any channel
        bright_mask = (r > 200) | (g > 200) | (b > 200)
        
        ball_mask = white_mask | bright_mask
        coords = np.argwhere(ball_mask)
        
        if len(coords) == 0 or len(coords) > 25:
            return None
            
        y_center = float(np.mean(coords[:, 0])) + 95
        x_center = float(np.mean(coords[:, 1])) + 8
        return (x_center, y_center)

    def _find_ball_method3(self, obs):
        """Edge-based detection for small moving objects."""
        field = obs[95:180, 8:152, :]
        gray = np.mean(field, axis=2)
        
        # Simple edge detection
        edges_y = np.abs(np.diff(gray, axis=0))
        edges_x = np.abs(np.diff(gray, axis=1))
        
        # Pad to maintain shape
        edges_y = np.pad(edges_y, ((0, 1), (0, 0)), mode='constant')
        edges_x = np.pad(edges_x, ((0, 0), (0, 1)), mode='constant')
        
        edge_strength = edges_y + edges_x
        strong_edges = edge_strength > 30
        
        coords = np.argwhere(strong_edges)
        
        if len(coords) == 0 or len(coords) > 15:
            return None
            
        y_center = float(np.mean(coords[:, 0])) + 95
        x_center = float(np.mean(coords[:, 1])) + 8
        return (x_center, y_center)

    def _update_ball_tracking(self, ball_x, ball_y):
        """Update ball position and velocity tracking."""
        current_pos = (ball_x, ball_y, self._steps)
        self._ball_positions.append(current_pos)
        if len(self._ball_positions) > 5:
            self._ball_positions.pop(0)

        # Calculate velocity with smoothing
        if len(self._ball_positions) >= 2:
            prev_pos = self._ball_positions[-2]
            dt = max(1, current_pos[2] - prev_pos[2])
            vx = (current_pos[0] - prev_pos[0]) / dt
            vy = (current_pos[1] - prev_pos[1]) / dt
            self._ball_velocity = (vx, vy)
            
            # Update consistent velocity with exponential smoothing
            alpha = 0.3
            self._consistent_velocity = (
                alpha * vx + (1 - alpha) * self._consistent_velocity[0],
                alpha * vy + (1 - alpha) * self._consistent_velocity[1]
            )
            
            # Update velocity confidence
            if len(self._ball_positions) >= 3:
                self._velocity_confidence = min(1.0, self._velocity_confidence + 0.2)
            
        self._last_ball_pos = current_pos

    def _calculate_target_position(self, ball_x, ball_y, paddle_x):
        """Calculate optimal target position."""
        if len(self._ball_positions) < 2:
            return ball_x
            
        vx, vy = self._consistent_velocity
        
        # Simple intercept calculation
        paddle_y = 189
        if ball_y >= paddle_y or abs(vy) < 0.1:
            return ball_x
            
        # Time to intercept
        time_to_intercept = (paddle_y - ball_y) / max(0.5, abs(vy))
        
        # Predicted position with wall bounces
        predicted_x = ball_x + vx * time_to_intercept
        
        # Handle wall bounces
        left_wall, right_wall = 8, 152
        bounce_count = 0
        
        while (predicted_x < left_wall or predicted_x > right_wall) and bounce_count < 2:
            bounce_count += 1
            if predicted_x < left_wall:
                predicted_x = left_wall + (left_wall - predicted_x)
                vx = -vx * 0.9  # Some energy loss
            elif predicted_x > right_wall:
                predicted_x = right_wall - (predicted_x - right_wall)
                vx = -vx * 0.9
        
        # Clamp to valid range
        predicted_x = max(left_wall + 5, min(right_wall - 5, predicted_x))
        
        # Add anticipation based on ball direction
        if abs(vx) > 1:
            anticipation = min(6, abs(vx) * 1.5) * (1 if vx > 0 else -1)
            predicted_x += anticipation * 0.4
            
        return predicted_x

    def _predict_from_last_known(self):
        """Predict ball position from last known state."""
        if self._last_ball_pos is None:
            return None
            
        last_x, last_y, last_step = self._last_ball_pos
        frames_elapsed = self._steps - last_step
        
        vx, vy = self._consistent_velocity
        estimated_x = last_x + vx * frames_elapsed
        estimated_y = last_y + vy * frames_elapsed
        
        if estimated_y > 200:  # Ball likely lost
            return None
            
        return self._calculate_target_position(estimated_x, estimated_y, self._last_paddle_x)

    def _move_towards_target(self, paddle_x, target_x, urgent=False):
        """Move paddle towards target position."""
        distance = target_x - paddle_x
        
        # Dynamic threshold based on urgency
        if urgent:
            threshold = 1.0
        else:
            threshold = 2.0
        
        if abs(distance) > threshold:
            if distance > 0:
                return self.right
            else:
                return self.left
        else:
            # Fine adjustment
            if abs(distance) > 0.5:
                return self.right if distance > 0 else self.left
            else:
                return self.noop

    def _find_paddle_x(self, obs):
        """Find paddle x-center with improved robustness."""
        # Look in multiple rows for paddle
        for row_start in [189, 188, 190]:
            if row_start + 3 < obs.shape[0]:
                paddle_region = obs[row_start:row_start+3, :, :]
                brightness = np.max(paddle_region, axis=2)
                bright_pixels = brightness > 140
                coords = np.argwhere(bright_pixels)
                
                if len(coords) > 5:  # Paddle should have multiple pixels
                    return float(np.mean(coords[:, 1]))
        
        # Fallback to last known position
        return self._last_paddle_x


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