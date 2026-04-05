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
    # Enhanced anticipatory positioning with velocity-aware momentum and improved edge case handling
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
        self._ball_velocity = (0, 0)
        self._velocity_history = []
        self._anticipation_offset = 0

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.
        
        Enhanced strategy with anticipatory positioning and velocity-aware movement.
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
            self._trajectory_confidence = max(0, self._trajectory_confidence - 0.3)
            
            # Emergency ball recovery
            if self._ball_lost_frames > 12:
                return self.fire
            
            # Use velocity-based prediction if available
            if len(self._velocity_history) >= 2 and self._trajectory_confidence > 0.2:
                predicted_x = self._predict_with_velocity()
                if predicted_x is not None:
                    return self._move_towards_target_enhanced(paddle_x, predicted_x, urgent=True)
            
            # Anticipatory positioning when ball is lost
            center_x = 80 + self._anticipation_offset
            return self._move_towards_target_enhanced(paddle_x, center_x, urgent=False)
        else:
            ball_x, ball_y = ball_pos
            self._no_ball_count = 0
            self._ball_lost_frames = 0
            self._trajectory_confidence = min(1.0, self._trajectory_confidence + 0.4)

        # Update ball position and velocity history
        self._update_ball_tracking(ball_x, ball_y)

        # Calculate target position with anticipatory offset
        target_x = self._calculate_enhanced_target_position(ball_x, ball_y)
        
        # Determine urgency and anticipation based on ball state
        urgent = ball_y > 65 or (abs(self._ball_velocity[1]) > 3)
        
        return self._move_towards_target_enhanced(paddle_x, target_x, urgent)

    def _update_ball_tracking(self, ball_x, ball_y):
        """Update ball position and velocity tracking."""
        self._ball_positions.append((ball_x, ball_y, self._steps))
        if len(self._ball_positions) > 6:
            self._ball_positions.pop(0)

        # Calculate current velocity
        if len(self._ball_positions) >= 2:
            prev_pos = self._ball_positions[-2]
            curr_pos = self._ball_positions[-1]
            dt = max(1, curr_pos[2] - prev_pos[2])
            vx = (curr_pos[0] - prev_pos[0]) / dt
            vy = (curr_pos[1] - prev_pos[1]) / dt
            self._ball_velocity = (vx, vy)
            
            # Track velocity history for stability
            self._velocity_history.append(self._ball_velocity)
            if len(self._velocity_history) > 4:
                self._velocity_history.pop(0)

    def _calculate_enhanced_target_position(self, ball_x, ball_y):
        """Calculate target position with enhanced anticipation."""
        if len(self._ball_positions) < 2:
            return ball_x
            
        # Get smoothed velocity
        if len(self._velocity_history) >= 2:
            recent_vx = [v[0] for v in self._velocity_history[-3:]]
            recent_vy = [v[1] for v in self._velocity_history[-3:]]
            smooth_vx = np.mean(recent_vx)
            smooth_vy = np.mean(recent_vy)
        else:
            smooth_vx, smooth_vy = self._ball_velocity

        # Enhanced intercept prediction
        predicted_x = self._predict_intercept_enhanced(ball_x, ball_y, smooth_vx, smooth_vy)
        
        # Add anticipatory offset based on ball velocity direction
        if abs(smooth_vx) > 1.0:
            anticipation_factor = min(8.0, abs(smooth_vx) * 2.0)
            if smooth_vx > 0:  # Ball moving right
                self._anticipation_offset = anticipation_factor
            else:  # Ball moving left
                self._anticipation_offset = -anticipation_factor
        else:
            self._anticipation_offset *= 0.8  # Decay offset
        
        return predicted_x + self._anticipation_offset * 0.3

    def _predict_intercept_enhanced(self, ball_x, ball_y, vx, vy):
        """Enhanced physics-based intercept prediction."""
        paddle_y = 189
        
        if ball_y >= paddle_y or abs(vy) < 0.1:
            return ball_x
            
        # Calculate time to intercept
        frames_to_paddle = (paddle_y - ball_y) / max(0.5, abs(vy))
        
        # Project position with wall bounces
        predicted_x = ball_x + vx * frames_to_paddle
        predicted_vx = vx
        
        # Simulate wall bounces with energy conservation
        bounce_count = 0
        left_wall, right_wall = 8, 152
        
        while (predicted_x < left_wall or predicted_x > right_wall) and bounce_count < 3:
            bounce_count += 1
            energy_loss = 0.95 - bounce_count * 0.05  # Gradual energy loss
            
            if predicted_x < left_wall:
                overshoot = left_wall - predicted_x
                predicted_x = left_wall + overshoot * energy_loss
                predicted_vx = -predicted_vx * energy_loss
            elif predicted_x > right_wall:
                overshoot = predicted_x - right_wall
                predicted_x = right_wall - overshoot * energy_loss
                predicted_vx = -predicted_vx * energy_loss
        
        return max(left_wall, min(right_wall, predicted_x))

    def _predict_with_velocity(self):
        """Predict ball position using velocity history when ball is lost."""
        if not self._velocity_history:
            return None
            
        # Use last known position and velocity
        if self._ball_positions:
            last_pos = self._ball_positions[-1]
            last_vx, last_vy = self._velocity_history[-1]
            
            # Estimate current position
            frames_elapsed = self._ball_lost_frames
            estimated_x = last_pos[0] + last_vx * frames_elapsed
            estimated_y = last_pos[1] + last_vy * frames_elapsed
            
            if estimated_y < 200:  # Ball still in play
                return self._predict_intercept_enhanced(estimated_x, estimated_y, last_vx, last_vy)
                
        return None

    def _move_towards_target_enhanced(self, paddle_x, target_x, urgent=False):
        """Enhanced movement with velocity-aware momentum."""
        distance = target_x - paddle_x
        
        # Dynamic threshold based on ball velocity and urgency
        base_threshold = 1.8
        velocity_factor = min(2.0, abs(self._ball_velocity[0]) * 0.3) if self._ball_velocity[0] != 0 else 1.0
        
        if urgent:
            threshold = base_threshold * 0.5 / velocity_factor
        else:
            threshold = base_threshold * (1.1 - self._trajectory_confidence * 0.3)
        
        # Enhanced momentum system
        if abs(distance) > threshold:
            desired_direction = 1 if distance > 0 else -1
            
            # Adjust momentum based on urgency and ball velocity
            if urgent and abs(self._ball_velocity[1]) > 2:
                momentum_factor = 0.8  # More aggressive when ball is fast
            elif urgent:
                momentum_factor = 0.6
            else:
                momentum_factor = 0.4
            
            self._action_momentum = momentum_factor * self._action_momentum + (1 - momentum_factor) * desired_direction
            
            # Dynamic movement threshold
            if urgent and abs(distance) > 5:
                move_threshold = 0.2  # Very responsive for urgent situations
            elif urgent:
                move_threshold = 0.3
            else:
                move_threshold = 0.4
            
            if self._action_momentum > move_threshold:
                return self.right
            elif self._action_momentum < -move_threshold:
                return self.left
            else:
                return self.noop
        else:
            # Fine positioning with velocity consideration
            self._action_momentum *= 0.7
            if abs(distance) > 0.8:
                return self.right if distance > 0 else self.left
            else:
                return self.noop

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