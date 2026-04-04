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
    # Enhanced paddle movement with adaptive speed and improved ball interception
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
        self._consecutive_moves = 0
        self._last_move_direction = None

    def act(self, obs):
        """
        Enhanced positioning strategy with adaptive paddle movement.
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
        
        # Move paddle with enhanced precision and speed
        return self._move_paddle_enhanced(target_x, ball_y)

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
        """Calculate optimal paddle target position with improved prediction."""
        # For balls high up or moving away, just track with slight anticipation
        if ball_y < 130:
            if len(self._ball_history) >= 2:
                last_x = self._ball_history[-2][0]
                movement = ball_x - last_x
                return ball_x + movement * 0.5
            return ball_x
            
        # Check if ball is moving toward paddle
        if self._ball_velocity is not None:
            vx, vy = self._ball_velocity
            
            # Only predict if ball is moving downward
            if vy > 0.2:
                # Time to reach paddle level with better accuracy
                paddle_y = 189
                time_to_paddle = max(1, (paddle_y - ball_y) / vy)
                
                # Predict impact position
                predicted_x = ball_x + vx * time_to_paddle
                
                # Handle wall bounces with improved accuracy
                left_wall, right_wall = 6, 154
                
                # Improved bounce simulation
                while predicted_x < left_wall or predicted_x > right_wall:
                    if predicted_x < left_wall:
                        predicted_x = left_wall + (left_wall - predicted_x)
                    elif predicted_x > right_wall:
                        predicted_x = right_wall - (predicted_x - right_wall)
                
                # Enhanced paddle positioning strategy
                # Position paddle for optimal ball control
                if abs(vx) > 0.3:
                    # For fast moving balls, position to hit with paddle edge for better angle control
                    if ball_y > 170:  # Very close to paddle
                        control_offset = min(8, abs(vx) * 2.0) * (1 if vx > 0 else -1)
                        predicted_x += control_offset * 0.4
                    else:
                        # Standard positioning with slight lead
                        control_offset = min(6, abs(vx) * 1.5) * (1 if vx > 0 else -1)
                        predicted_x += control_offset * 0.3
                
                # Ensure target is reachable
                predicted_x = max(12, min(148, predicted_x))
                return predicted_x
        
        # Fallback: track ball with enhanced anticipation
        if len(self._ball_history) >= 2:
            last_x = self._ball_history[-2][0]
            movement = ball_x - last_x
            # Increase anticipation based on ball proximity
            anticipation = 0.5 if ball_y > 160 else 0.3
            return ball_x + movement * anticipation
        
        return ball_x

    def _move_paddle_enhanced(self, target_x, ball_y):
        """Enhanced paddle movement with adaptive speed and momentum."""
        distance = target_x - self._paddle_x
        
        # Calculate urgency and required precision based on ball position
        if ball_y > 175:
            # Ball is very close - maximum urgency, tight control
            move_threshold = 0.8
            precision_mode = True
        elif ball_y > 160:
            # Ball is close - high urgency
            move_threshold = 1.5
            precision_mode = True
        elif ball_y > 140:
            # Ball is approaching - moderate urgency
            move_threshold = 2.5
            precision_mode = False
        else:
            # Ball is far - be selective
            move_threshold = 4.0
            precision_mode = False
        
        # Enhanced movement logic
        if abs(distance) > move_threshold:
            direction = self.right if distance > 0 else self.left
            
            # Track consecutive moves for momentum
            if direction == self._last_move_direction:
                self._consecutive_moves += 1
            else:
                self._consecutive_moves = 1
                self._last_move_direction = direction
            
            # In precision mode with large distances, allow faster movement
            if precision_mode and abs(distance) > 8:
                # Continue moving aggressively when far from target in critical situations
                return direction
            elif precision_mode and abs(distance) > 3:
                # Moderate movement when getting closer
                return direction
            elif not precision_mode or abs(distance) > move_threshold:
                # Normal movement
                return direction
            else:
                # Fine positioning - occasionally skip moves for precision
                if self._consecutive_moves % 3 == 0 and abs(distance) < 2:
                    return self.noop
                return direction
        else:
            # Reset movement tracking when stopped
            self._consecutive_moves = 0
            self._last_move_direction = None
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