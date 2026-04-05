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
    # Simplified but more aggressive ball tracking with dynamic paddle speed control
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
        self._paddle_x = 80.0
        self._ball_lost_count = 0
        self._last_ball_y = 0
        self._ball_moving_down = False
        self._emergency_mode = False

    def act(self, obs):
        """
        Simplified but aggressive ball tracking with dynamic response.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Find ball and paddle
        ball_pos = self._find_ball_robust(obs)
        self._paddle_x = self._find_paddle_x(obs)

        if ball_pos is None:
            self._ball_lost_count += 1
            self._emergency_mode = True
            
            # Emergency: fire if ball lost too long
            if self._ball_lost_count > 20:
                return self.fire
                
            # Move to center when ball is lost
            if self._paddle_x < 75:
                return self.right
            elif self._paddle_x > 85:
                return self.left
            else:
                return self.noop
        else:
            ball_x, ball_y = ball_pos
            self._ball_lost_count = 0
            self._emergency_mode = False

        # Track ball movement
        self._update_ball_tracking(ball_x, ball_y)
        
        # Determine if ball is moving toward paddle
        self._ball_moving_down = ball_y > self._last_ball_y
        self._last_ball_y = ball_y

        # Calculate target position with prediction
        target_x = self._calculate_target_aggressive(ball_x, ball_y)
        
        # Dynamic paddle control based on urgency
        return self._move_paddle_dynamic(target_x, ball_y)

    def _find_ball_robust(self, obs):
        """Robust ball detection using multiple criteria."""
        # Focus on game field
        field = obs[95:185, 8:152, :]
        
        # Method 1: Bright white pixels (most reliable)
        brightness = np.max(field, axis=2)
        bright_mask = brightness > 180
        
        # Method 2: Pure white pixels
        white_mask = np.all(field > [200, 200, 200], axis=2)
        
        # Method 3: High contrast pixels
        gray = np.mean(field, axis=2)
        contrast = np.max(field, axis=2) - np.min(field, axis=2)
        contrast_mask = contrast > 100
        
        # Combine methods
        ball_mask = bright_mask | white_mask | (contrast_mask & (brightness > 150))
        
        coords = np.argwhere(ball_mask)
        
        # Filter out large regions (likely not the ball)
        if len(coords) == 0 or len(coords) > 30:
            return None
            
        # Return center of detected region
        y_center = float(np.mean(coords[:, 0])) + 95
        x_center = float(np.mean(coords[:, 1])) + 8
        
        # Sanity check - ball should be in reasonable position
        if y_center < 95 or y_center > 185 or x_center < 8 or x_center > 152:
            return None
            
        return (x_center, y_center)

    def _update_ball_tracking(self, ball_x, ball_y):
        """Update ball position history for velocity calculation."""
        self._ball_history.append((ball_x, ball_y, self._steps))
        
        # Keep only recent history
        if len(self._ball_history) > 8:
            self._ball_history.pop(0)

    def _calculate_target_aggressive(self, ball_x, ball_y):
        """Calculate target position with aggressive prediction."""
        # If ball is high up, just track it
        if ball_y < 120:
            return ball_x
            
        # Calculate velocity if we have history
        if len(self._ball_history) >= 3:
            recent = self._ball_history[-3:]
            vx = (recent[-1][0] - recent[0][0]) / max(1, recent[-1][2] - recent[0][2])
            vy = (recent[-1][1] - recent[0][1]) / max(1, recent[-1][2] - recent[0][2])
        else:
            return ball_x
            
        # Only predict if ball is moving down
        if vy <= 0:
            return ball_x
            
        # Predict where ball will be at paddle level
        paddle_y = 189
        time_to_paddle = (paddle_y - ball_y) / max(0.5, vy)
        
        predicted_x = ball_x + vx * time_to_paddle
        
        # Handle wall bounces
        left_wall, right_wall = 8, 152
        
        # Simple bounce calculation
        if predicted_x < left_wall:
            predicted_x = left_wall + (left_wall - predicted_x)
        elif predicted_x > right_wall:
            predicted_x = right_wall - (predicted_x - right_wall)
            
        # Add lead based on ball speed and direction
        if abs(vx) > 1:
            lead = min(8, abs(vx) * 2) * (1 if vx > 0 else -1)
            predicted_x += lead * 0.6
            
        # Clamp to playfield
        predicted_x = max(15, min(145, predicted_x))
        
        return predicted_x

    def _move_paddle_dynamic(self, target_x, ball_y):
        """Dynamic paddle movement based on urgency."""
        distance = target_x - self._paddle_x
        
        # Determine urgency based on ball position and movement
        urgent = (ball_y > 160) or (self._ball_moving_down and ball_y > 140)
        very_urgent = ball_y > 175
        
        # Dynamic thresholds
        if very_urgent:
            threshold = 0.5
        elif urgent:
            threshold = 1.5
        else:
            threshold = 3.0
            
        # More aggressive movement when urgent
        if abs(distance) > threshold:
            if distance > 0:
                return self.right
            else:
                return self.left
        else:
            # Fine positioning
            if abs(distance) > 0.8:
                return self.right if distance > 0 else self.left
            else:
                return self.noop

    def _find_paddle_x(self, obs):
        """Find paddle x-center position."""
        # Look at bottom rows where paddle should be
        for row in [189, 188, 190, 187]:
            if row < obs.shape[0]:
                paddle_row = obs[row, :, :]
                brightness = np.max(paddle_row, axis=1)
                bright_pixels = brightness > 140
                
                # Find continuous bright region (paddle)
                coords = np.where(bright_pixels)[0]
                if len(coords) > 8:  # Paddle should be reasonably wide
                    return float(np.mean(coords))
        
        # Fallback: return current position
        return self._paddle_x


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