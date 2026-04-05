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
    # Simplified precise positioning with optimal paddle placement strategy
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

    def act(self, obs):
        """
        Simplified precise positioning strategy.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            self._last_fire_time = self._steps
            return self.fire

        # Find ball and paddle
        ball_pos = self._find_ball_robust(obs)
        self._paddle_x = self._find_paddle_x(obs)

        if ball_pos is None:
            self._ball_lost_count += 1
            
            # Re-fire if ball lost for too long
            if self._ball_lost_count > 25 or (self._ball_lost_count > 12 and self._steps - self._last_fire_time > 60):
                self._last_fire_time = self._steps
                self._fired = False
                return self.fire
                
            # Center paddle when ball is lost
            center_target = 80
            if abs(self._paddle_x - center_target) > 2:
                return self.right if self._paddle_x < center_target else self.left
            else:
                return self.noop
        else:
            ball_x, ball_y = ball_pos
            self._ball_lost_count = 0

        # Update ball tracking
        self._update_ball_tracking(ball_x, ball_y)
        
        # Calculate optimal target position
        target_x = self._calculate_optimal_target(ball_x, ball_y)
        
        # Move paddle with precision
        return self._move_paddle_precise(target_x, ball_y)

    def _find_ball_robust(self, obs):
        """Robust ball detection with multiple fallback methods."""
        # Focus on main game area
        field = obs[93:187, 6:154, :]
        
        # Method 1: Very bright pixels (primary)
        brightness = np.max(field, axis=2)
        very_bright = brightness > 200
        
        # Method 2: Pure white detection
        white_pixels = np.all(field >= [210, 210, 210], axis=2)
        
        # Method 3: High contrast small objects
        contrast = np.max(field, axis=2) - np.min(field, axis=2)
        high_contrast = (contrast > 120) & (brightness > 160)
        
        # Combine detection methods
        ball_candidates = very_bright | white_pixels | high_contrast
        
        coords = np.argwhere(ball_candidates)
        
        # Filter by size - ball should be small
        if len(coords) == 0 or len(coords) > 25:
            return None
            
        # Get center position
        y_center = float(np.mean(coords[:, 0])) + 93
        x_center = float(np.mean(coords[:, 1])) + 6
        
        # Validate position
        if y_center < 93 or y_center > 187 or x_center < 6 or x_center > 154:
            return None
            
        return (x_center, y_center)

    def _update_ball_tracking(self, ball_x, ball_y):
        """Update ball position history and calculate velocity."""
        self._ball_history.append((ball_x, ball_y, self._steps))
        
        # Keep reasonable history
        if len(self._ball_history) > 6:
            self._ball_history.pop(0)
            
        # Calculate velocity from recent positions
        if len(self._ball_history) >= 3:
            recent = self._ball_history[-3:]
            
            # Calculate velocity over the span
            time_span = recent[-1][2] - recent[0][2]
            if time_span > 0:
                vx = (recent[-1][0] - recent[0][0]) / time_span
                vy = (recent[-1][1] - recent[0][1]) / time_span
                
                # Store velocity sample if reasonable
                if abs(vx) < 6 and abs(vy) < 6:
                    self._velocity_samples.append((vx, vy))
                    
                    # Keep only recent samples
                    if len(self._velocity_samples) > 5:
                        self._velocity_samples.pop(0)
                    
                    # Calculate stable velocity from samples
                    if len(self._velocity_samples) >= 2:
                        vx_vals = [v[0] for v in self._velocity_samples]
                        vy_vals = [v[1] for v in self._velocity_samples]
                        self._ball_velocity = (np.mean(vx_vals), np.mean(vy_vals))

    def _calculate_optimal_target(self, ball_x, ball_y):
        """Calculate optimal paddle target position."""
        # For balls high up or moving away, just track
        if ball_y < 130:
            return ball_x
            
        # Check if ball is moving toward paddle
        if self._ball_velocity is not None:
            vx, vy = self._ball_velocity
            
            # Only predict if ball is moving downward
            if vy > 0.3:
                # Time to reach paddle level
                paddle_y = 189
                time_to_paddle = (paddle_y - ball_y) / vy
                
                # Predict impact position
                predicted_x = ball_x + vx * time_to_paddle
                
                # Handle wall bounces with improved accuracy
                left_wall, right_wall = 6, 154
                
                # Simple bounce simulation
                if predicted_x < left_wall:
                    predicted_x = left_wall + (left_wall - predicted_x)
                elif predicted_x > right_wall:
                    predicted_x = right_wall - (predicted_x - right_wall)
                
                # Optimal paddle positioning strategy
                # Position paddle center slightly ahead of predicted impact for better control
                if abs(vx) > 0.5:
                    # For moving balls, position to hit with optimal part of paddle
                    optimal_offset = min(6, abs(vx) * 1.5) * (1 if vx > 0 else -1)
                    predicted_x += optimal_offset * 0.3
                
                # Ensure target is reachable
                predicted_x = max(12, min(148, predicted_x))
                return predicted_x
        
        # Fallback: track ball with minimal anticipation
        if len(self._ball_history) >= 2:
            last_x = self._ball_history[-2][0]
            movement = ball_x - last_x
            return ball_x + movement * 0.3
        
        return ball_x

    def _move_paddle_precise(self, target_x, ball_y):
        """Precise paddle movement with distance-based thresholds."""
        distance = target_x - self._paddle_x
        
        # Calculate movement urgency based on ball height
        if ball_y > 160:
            # Ball is close - be more aggressive
            move_threshold = 1.5
        elif ball_y > 140:
            # Ball is approaching - moderate urgency
            move_threshold = 3.0
        else:
            # Ball is far - be more selective
            move_threshold = 4.5
        
        # Move if beyond threshold
        if abs(distance) > move_threshold:
            return self.right if distance > 0 else self.left
        else:
            return self.noop

    def _find_paddle_x(self, obs):
        """Find paddle center position with improved detection."""
        # Check multiple rows for paddle
        paddle_rows = [189, 188, 190, 187, 191]
        
        for row in paddle_rows:
            if row < obs.shape[0]:
                paddle_row = obs[row, :, :]
                
                # Look for bright pixels
                brightness = np.max(paddle_row, axis=1)
                paddle_pixels = brightness > 130
                
                # Find continuous regions
                coords = np.where(paddle_pixels)[0]
                if len(coords) >= 10:  # Paddle should be reasonably wide
                    # Use median for more stable center detection
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