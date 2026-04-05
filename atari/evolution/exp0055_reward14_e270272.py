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
    # Enhanced velocity prediction with multi-frame smoothing and adaptive positioning
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
        self._velocity_history = []
        self._consistent_velocity = None

    def act(self, obs):
        """
        Enhanced ball tracking with improved velocity prediction.
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
            if self._ball_lost_count > 15:
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

        # Track ball movement with enhanced velocity calculation
        self._update_ball_tracking_enhanced(ball_x, ball_y)
        
        # Determine if ball is moving toward paddle
        self._ball_moving_down = ball_y > self._last_ball_y
        self._last_ball_y = ball_y

        # Calculate target position with enhanced prediction
        target_x = self._calculate_target_enhanced(ball_x, ball_y)
        
        # Dynamic paddle control based on urgency
        return self._move_paddle_adaptive(target_x, ball_y, ball_x)

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

    def _update_ball_tracking_enhanced(self, ball_x, ball_y):
        """Enhanced ball position history with velocity smoothing."""
        self._ball_history.append((ball_x, ball_y, self._steps))
        
        # Keep only recent history
        if len(self._ball_history) > 10:
            self._ball_history.pop(0)
            
        # Calculate and smooth velocity
        if len(self._ball_history) >= 4:
            # Use multiple frame pairs for velocity calculation
            velocities = []
            for i in range(len(self._ball_history) - 3, len(self._ball_history)):
                if i >= 3:
                    dt = max(1, self._ball_history[i][2] - self._ball_history[i-3][2])
                    vx = (self._ball_history[i][0] - self._ball_history[i-3][0]) / dt
                    vy = (self._ball_history[i][1] - self._ball_history[i-3][1]) / dt
                    velocities.append((vx, vy))
            
            if velocities:
                # Average recent velocities for smoothing
                avg_vx = np.mean([v[0] for v in velocities])
                avg_vy = np.mean([v[1] for v in velocities])
                
                self._velocity_history.append((avg_vx, avg_vy))
                if len(self._velocity_history) > 5:
                    self._velocity_history.pop(0)
                    
                # Check for consistent velocity pattern
                if len(self._velocity_history) >= 3:
                    recent_vx = [v[0] for v in self._velocity_history[-3:]]
                    recent_vy = [v[1] for v in self._velocity_history[-3:]]
                    
                    if np.std(recent_vx) < 0.5 and np.std(recent_vy) < 0.5:
                        self._consistent_velocity = (np.mean(recent_vx), np.mean(recent_vy))
                    else:
                        self._consistent_velocity = None

    def _calculate_target_enhanced(self, ball_x, ball_y):
        """Enhanced target calculation with better velocity prediction."""
        # If ball is high up, just track it
        if ball_y < 115:
            return ball_x
            
        # Use consistent velocity if available, otherwise calculate fresh
        if self._consistent_velocity is not None:
            vx, vy = self._consistent_velocity
        elif len(self._ball_history) >= 4:
            recent = self._ball_history[-4:]
            dt = max(1, recent[-1][2] - recent[0][2])
            vx = (recent[-1][0] - recent[0][0]) / dt
            vy = (recent[-1][1] - recent[0][1]) / dt
        else:
            return ball_x
            
        # Only predict if ball is moving down significantly
        if vy <= 0.3:
            return ball_x
            
        # Predict where ball will be at paddle level
        paddle_y = 189
        time_to_paddle = (paddle_y - ball_y) / max(0.5, vy)
        
        predicted_x = ball_x + vx * time_to_paddle
        
        # Enhanced wall bounce calculation
        left_wall, right_wall = 8, 152
        
        # Handle multiple bounces if ball is moving very fast
        while predicted_x < left_wall or predicted_x > right_wall:
            if predicted_x < left_wall:
                predicted_x = left_wall + (left_wall - predicted_x)
                vx = -vx  # Reverse velocity after bounce
            elif predicted_x > right_wall:
                predicted_x = right_wall - (predicted_x - right_wall)
                vx = -vx  # Reverse velocity after bounce
                
        # Adaptive lead based on ball speed and consistency
        if abs(vx) > 0.8:
            confidence = 1.0 if self._consistent_velocity else 0.6
            lead_factor = min(10, abs(vx) * 3) * confidence
            lead = lead_factor * (1 if vx > 0 else -1)
            predicted_x += lead * 0.4
            
        # Smart positioning based on ball trajectory
        if ball_y > 150:  # Ball is close, be more precise
            predicted_x = max(20, min(140, predicted_x))
        else:  # Ball is farther, allow wider range
            predicted_x = max(15, min(145, predicted_x))
        
        return predicted_x

    def _move_paddle_adaptive(self, target_x, ball_y, ball_x):
        """Adaptive paddle movement with context-aware thresholds."""
        distance = target_x - self._paddle_x
        
        # Calculate urgency with multiple factors
        y_urgency = max(0, (ball_y - 140) / 45)  # 0 to 1 based on y position
        
        # Add horizontal urgency if ball is moving away
        h_urgency = 0
        if len(self._ball_history) >= 2:
            last_x = self._ball_history[-2][0]
            if abs(ball_x - self._paddle_x) > abs(last_x - self._paddle_x):
                h_urgency = 0.3
        
        total_urgency = min(1.0, y_urgency + h_urgency)
        
        # Dynamic thresholds based on urgency
        base_threshold = 4.0 - 3.5 * total_urgency  # 4.0 when calm, 0.5 when urgent
        fine_threshold = base_threshold * 0.3
        
        # Movement decision with adaptive sensitivity
        if abs(distance) > base_threshold:
            return self.right if distance > 0 else self.left
        elif abs(distance) > fine_threshold:
            # Probability-based fine movement for smoother control
            move_prob = abs(distance) / base_threshold
            if move_prob > 0.4:  # Move if distance is significant enough
                return self.right if distance > 0 else self.left
            else:
                return self.noop
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