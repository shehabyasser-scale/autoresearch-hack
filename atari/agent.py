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
    # Ball state-aware agent with simplified but more reliable prediction and improved emergency handling
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
        self._ball_state = 'unknown'  # 'approaching', 'receding', 'unknown'
        self._stable_velocity = None
        self._velocity_confidence = 0
        self._last_fire_time = 0

    def act(self, obs):
        """
        Ball state-aware tracking with simplified prediction.
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
            
            # More aggressive re-firing if ball lost for too long
            if self._ball_lost_count > 20 or (self._ball_lost_count > 10 and self._steps - self._last_fire_time > 50):
                self._last_fire_time = self._steps
                self._fired = False  # Reset fired state
                return self.fire
                
            # Smart centering when ball is lost
            center_target = 80
            if abs(self._paddle_x - center_target) > 3:
                return self.right if self._paddle_x < center_target else self.left
            else:
                return self.noop
        else:
            ball_x, ball_y = ball_pos
            self._ball_lost_count = 0

        # Update ball tracking and determine state
        self._update_ball_state(ball_x, ball_y)
        
        # Calculate target position based on ball state
        target_x = self._calculate_smart_target(ball_x, ball_y)
        
        # Move paddle with state-aware logic
        action = self._move_paddle_smart(target_x, ball_y, ball_x)
        self._last_action = action
        return action

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

    def _update_ball_state(self, ball_x, ball_y):
        """Update ball position history and determine movement state."""
        self._ball_history.append((ball_x, ball_y, self._steps))
        
        # Keep reasonable history
        if len(self._ball_history) > 8:
            self._ball_history.pop(0)
            
        # Determine ball state with improved logic
        if len(self._ball_history) >= 4:
            recent_positions = self._ball_history[-4:]
            
            # Calculate y-direction movement
            y_changes = []
            for i in range(1, len(recent_positions)):
                y_changes.append(recent_positions[i][1] - recent_positions[i-1][1])
            
            avg_y_change = np.mean(y_changes) if y_changes else 0
            
            # Determine state based on consistent y movement
            if avg_y_change > 0.5:
                self._ball_state = 'approaching'
            elif avg_y_change < -0.5:
                self._ball_state = 'receding'
            else:
                self._ball_state = 'unknown'
                
            # Calculate stable velocity for approaching balls
            if self._ball_state == 'approaching' and len(self._ball_history) >= 3:
                recent = self._ball_history[-3:]
                time_span = recent[-1][2] - recent[0][2]
                if time_span > 0:
                    vx = (recent[-1][0] - recent[0][0]) / time_span
                    vy = (recent[-1][1] - recent[0][1]) / time_span
                    
                    # Only update if velocity seems reasonable
                    if abs(vx) < 5 and 0.3 < vy < 5:
                        if self._stable_velocity is None:
                            self._stable_velocity = (vx, vy)
                            self._velocity_confidence = 1
                        else:
                            # Smooth velocity updates
                            old_vx, old_vy = self._stable_velocity
                            alpha = 0.3
                            new_vx = alpha * vx + (1 - alpha) * old_vx
                            new_vy = alpha * vy + (1 - alpha) * old_vy
                            self._stable_velocity = (new_vx, new_vy)
                            self._velocity_confidence = min(5, self._velocity_confidence + 1)
            else:
                # Decay confidence when ball is not approaching
                if self._velocity_confidence > 0:
                    self._velocity_confidence -= 1
                if self._velocity_confidence <= 0:
                    self._stable_velocity = None

    def _calculate_smart_target(self, ball_x, ball_y):
        """Calculate target position based on ball state."""
        # For high balls or receding balls, just track
        if ball_y < 120 or self._ball_state == 'receding':
            return ball_x
            
        # For approaching balls, use prediction if available
        if (self._ball_state == 'approaching' and 
            self._stable_velocity is not None and 
            self._velocity_confidence >= 2):
            
            vx, vy = self._stable_velocity
            
            # Only predict if ball is moving down reasonably fast
            if vy > 0.5:
                # Time to reach paddle level
                paddle_y = 189
                time_to_paddle = (paddle_y - ball_y) / vy
                
                # Predict x position
                predicted_x = ball_x + vx * time_to_paddle
                
                # Handle wall bounces (simplified)
                left_wall, right_wall = 6, 154
                if predicted_x < left_wall:
                    predicted_x = left_wall + (left_wall - predicted_x)
                elif predicted_x > right_wall:
                    predicted_x = right_wall - (predicted_x - right_wall)
                
                # Add smart positioning offset based on velocity
                if abs(vx) > 1.0:
                    # For fast horizontal movement, position slightly ahead
                    offset = min(8, abs(vx) * 2) * (1 if vx > 0 else -1)
                    predicted_x += offset * 0.5
                
                # Ensure target is within paddle reach
                predicted_x = max(15, min(145, predicted_x))
                return predicted_x
        
        # Fallback: track ball directly with slight anticipation
        if len(self._ball_history) >= 2:
            last_x = self._ball_history[-2][0]
            movement = ball_x - last_x
            # Small anticipation for direct tracking
            return ball_x + movement * 0.5
        
        return ball_x

    def _move_paddle_smart(self, target_x, ball_y, ball_x):
        """Smart paddle movement with state awareness."""
        distance = target_x - self._paddle_x
        
        # Calculate urgency based on multiple factors
        y_urgency = 0
        if ball_y > 130:
            y_urgency = (ball_y - 130) / 55  # 0 to 1
            
        state_urgency = 0
        if self._ball_state == 'approaching':
            state_urgency = 0.4
            
        # Add urgency if we're moving away from the ball
        direction_urgency = 0
        if self._last_action in [self.left, self.right]:
            if (self._last_action == self.left and ball_x > self._paddle_x) or \
               (self._last_action == self.right and ball_x < self._paddle_x):
                direction_urgency = 0.3
        
        total_urgency = min(1.0, y_urgency + state_urgency + direction_urgency)
        
        # Adaptive movement thresholds
        move_threshold = 5.0 - 4.0 * total_urgency  # 5.0 when calm, 1.0 when urgent
        fine_threshold = move_threshold * 0.4
        
        # Movement decision
        if abs(distance) > move_threshold:
            return self.right if distance > 0 else self.left
        elif abs(distance) > fine_threshold and total_urgency > 0.3:
            # Fine movement when somewhat urgent
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