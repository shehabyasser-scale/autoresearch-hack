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
    # Improved ball detection with simplified movement logic and better edge case handling
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
        self._last_action = self.noop
        self._ball_velocity = None
        self._velocity_samples = []
        self._last_fire_time = 0
        self._stable_ball_pos = None
        self._detection_confidence = 0

    def act(self, obs):
        """
        Enhanced positioning strategy with improved ball detection and simplified movement.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            self._last_fire_time = self._steps
            return self.fire

        # Find ball and paddle with improved detection
        ball_pos = self._find_ball_enhanced(obs)
        self._paddle_x = self._find_paddle_x(obs)

        if ball_pos is None:
            self._ball_lost_count += 1
            self._detection_confidence = max(0, self._detection_confidence - 1)
            
            # Re-fire if ball lost for too long
            if self._ball_lost_count > 20 or (self._ball_lost_count > 10 and self._steps - self._last_fire_time > 50):
                self._last_fire_time = self._steps
                self._fired = False
                return self.fire
                
            # Use last known stable position if available
            if self._stable_ball_pos is not None and self._ball_lost_count < 8:
                ball_x, ball_y = self._stable_ball_pos
                target_x = self._calculate_target_position(ball_x, ball_y)
                return self._move_paddle_smooth(target_x, ball_y)
                
            # Center paddle when ball is lost
            center_target = 80
            if abs(self._paddle_x - center_target) > 3:
                return self.right if self._paddle_x < center_target else self.left
            else:
                return self.noop
        else:
            ball_x, ball_y = ball_pos
            self._ball_lost_count = 0
            self._detection_confidence = min(10, self._detection_confidence + 1)
            
            # Update stable position only when confident
            if self._detection_confidence >= 3:
                self._stable_ball_pos = (ball_x, ball_y)

        # Update ball tracking
        self._update_ball_tracking(ball_x, ball_y)
        
        # Calculate optimal target position
        target_x = self._calculate_target_position(ball_x, ball_y)
        
        # Move paddle with smooth movement
        return self._move_paddle_smooth(target_x, ball_y)

    def _find_ball_enhanced(self, obs):
        """Enhanced ball detection with multiple methods and confidence scoring."""
        # Focus on main game area
        field = obs[93:187, 6:154, :]
        
        # Method 1: Bright white pixels (most reliable)
        white_mask = np.all(field >= [200, 200, 200], axis=2)
        
        # Method 2: High brightness
        brightness = np.max(field, axis=2)
        bright_mask = brightness >= 180
        
        # Method 3: Orange-ish ball color (backup)
        orange_mask = (field[:, :, 0] > 200) & (field[:, :, 1] > 100) & (field[:, :, 1] < 180) & (field[:, :, 2] < 100)
        
        # Combine masks with priority
        ball_mask = white_mask | (bright_mask & (brightness > 190)) | orange_mask
        
        coords = np.argwhere(ball_mask)
        
        # Filter by reasonable size (ball should be small cluster)
        if len(coords) == 0 or len(coords) > 30:
            return None
        
        # Remove outliers for better center calculation
        if len(coords) > 4:
            y_coords = coords[:, 0]
            x_coords = coords[:, 1]
            
            # Remove extreme outliers
            y_median = np.median(y_coords)
            x_median = np.median(x_coords)
            
            distances = np.sqrt((y_coords - y_median)**2 + (x_coords - x_median)**2)
            valid_mask = distances < np.percentile(distances, 80)
            coords = coords[valid_mask]
            
            if len(coords) == 0:
                return None
        
        # Calculate center
        y_center = float(np.mean(coords[:, 0])) + 93
        x_center = float(np.mean(coords[:, 1])) + 6
        
        # Validate position is in game area
        if y_center < 93 or y_center > 187 or x_center < 6 or x_center > 154:
            return None
            
        return (x_center, y_center)

    def _update_ball_tracking(self, ball_x, ball_y):
        """Update ball position history and calculate stable velocity."""
        self._ball_history.append((ball_x, ball_y, self._steps))
        
        # Keep reasonable history
        if len(self._ball_history) > 5:
            self._ball_history.pop(0)
            
        # Calculate velocity from stable positions
        if len(self._ball_history) >= 3:
            # Use first and last position for more stable velocity
            first = self._ball_history[0]
            last = self._ball_history[-1]
            
            time_span = last[2] - first[2]
            if time_span > 0:
                vx = (last[0] - first[0]) / time_span
                vy = (last[1] - first[1]) / time_span
                
                # Store velocity if reasonable
                if abs(vx) < 5 and abs(vy) < 5:
                    self._velocity_samples.append((vx, vy))
                    
                    # Keep recent samples
                    if len(self._velocity_samples) > 4:
                        self._velocity_samples.pop(0)
                    
                    # Calculate stable velocity
                    if len(self._velocity_samples) >= 2:
                        vx_vals = [v[0] for v in self._velocity_samples]
                        vy_vals = [v[1] for v in self._velocity_samples]
                        self._ball_velocity = (np.median(vx_vals), np.median(vy_vals))

    def _calculate_target_position(self, ball_x, ball_y):
        """Calculate target position with improved prediction accuracy."""
        # For high balls, use simple tracking
        if ball_y < 125:
            return ball_x
            
        # Use velocity prediction for approaching balls
        if self._ball_velocity is not None:
            vx, vy = self._ball_velocity
            
            # Only predict if ball is moving toward paddle
            if vy > 0.1:
                # Calculate time to reach paddle
                paddle_y = 189
                time_to_paddle = max(1, (paddle_y - ball_y) / vy)
                
                # Predict impact position
                predicted_x = ball_x + vx * time_to_paddle
                
                # Handle wall bounces
                left_wall, right_wall = 6, 154
                
                # Simple bounce calculation
                while predicted_x < left_wall or predicted_x > right_wall:
                    if predicted_x < left_wall:
                        predicted_x = left_wall + (left_wall - predicted_x)
                    elif predicted_x > right_wall:
                        predicted_x = right_wall - (predicted_x - right_wall)
                
                # Add strategic positioning offset based on ball speed
                if ball_y > 165:  # Close to paddle
                    if abs(vx) > 0.5:
                        # Position to hit ball with slight angle for control
                        offset = min(4, abs(vx) * 1.5) * (1 if vx > 0 else -1)
                        predicted_x += offset * 0.3
                
                # Ensure target is reachable
                predicted_x = max(10, min(150, predicted_x))
                return predicted_x
        
        # Fallback: simple tracking with slight anticipation
        if len(self._ball_history) >= 2:
            prev_x = self._ball_history[-2][0]
            movement = ball_x - prev_x
            anticipation = 0.4 if ball_y > 155 else 0.2
            return ball_x + movement * anticipation
        
        return ball_x

    def _move_paddle_smooth(self, target_x, ball_y):
        """Smooth paddle movement with distance-based speed control."""
        distance = target_x - self._paddle_x
        
        # Calculate urgency based on ball position
        if ball_y > 170:
            # Critical - ball very close
            threshold = 1.0
        elif ball_y > 150:
            # Important - ball approaching
            threshold = 2.0
        elif ball_y > 130:
            # Moderate - ball in mid area
            threshold = 3.5
        else:
            # Low priority - ball far away
            threshold = 5.0
        
        # Move if distance exceeds threshold
        if abs(distance) > threshold:
            return self.right if distance > 0 else self.left
        else:
            return self.noop

    def _find_paddle_x(self, obs):
        """Find paddle center position with multi-row detection."""
        # Check multiple rows around paddle area
        paddle_rows = [189, 188, 190, 187]
        
        for row in paddle_rows:
            if row < obs.shape[0]:
                paddle_row = obs[row, :, :]
                
                # Look for paddle color (bright pixels)
                brightness = np.max(paddle_row, axis=1)
                paddle_mask = brightness > 120
                
                coords = np.where(paddle_mask)[0]
                if len(coords) >= 8:  # Paddle should be reasonably wide
                    # Use median for stable center
                    return float(np.median(coords))
        
        # Fallback: return previous position
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