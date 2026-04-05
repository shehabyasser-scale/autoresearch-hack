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
    # Enhanced ball detection with multi-frame tracking and improved edge case handling
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
        self._frame_buffer = []
        self._last_known_ball_x = None
        self._emergency_mode = False

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.

        Enhanced strategy with multi-frame tracking and emergency mode.
        """
        self._steps += 1

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Store frame for multi-frame analysis
        self._frame_buffer.append(obs.copy())
        if len(self._frame_buffer) > 3:
            self._frame_buffer.pop(0)

        # Find ball position using enhanced detection with multi-frame support
        ball_x = self._find_ball_x_multiframe(obs)
        
        if ball_x is None:
            self._no_ball_count += 1
            self._confidence_score = max(0, self._confidence_score - 0.2)
            
            # Enhanced prediction fallback
            if len(self._ball_history) >= 3 and self._predicted_ball_x is not None and self._confidence_score > 0.15:
                ball_x = self._predicted_ball_x
                # Apply bounds checking
                ball_x = max(8, min(152, ball_x))
                
                # Decay prediction with distance from last known position
                if self._no_ball_count > 4:
                    if self._last_known_ball_x is not None:
                        # Blend prediction with last known position
                        blend_factor = min(0.7, self._no_ball_count * 0.1)
                        ball_x = ball_x * (1 - blend_factor) + self._last_known_ball_x * blend_factor
            
            if ball_x is None:
                # Enter emergency mode after extended ball loss
                if self._no_ball_count > 15:
                    self._emergency_mode = True
                    self._no_ball_count = 0
                    self._paddle_momentum = 0
                    return self.fire
                
                # Emergency positioning - move toward center if too far out
                paddle_x = self._find_paddle_x(obs)
                center_x = 80
                if abs(paddle_x - center_x) > 30:
                    return self.right if center_x > paddle_x else self.left
                
                # Continue momentum-based movement with decay
                if abs(self._paddle_momentum) > 0.3:
                    self._paddle_momentum *= 0.85
                    return self.right if self._paddle_momentum > 0 else self.left
                return self.noop
        else:
            self._no_ball_count = 0
            self._confidence_score = min(1.0, self._confidence_score + 0.5)
            self._last_known_ball_x = ball_x
            self._emergency_mode = False

        # Update ball history and compute enhanced velocity
        self._ball_history.append(ball_x)
        if len(self._ball_history) > 8:
            self._ball_history.pop(0)

        # Enhanced velocity computation with outlier filtering
        velocity = 0
        if len(self._ball_history) >= 4:
            positions = np.array(self._ball_history[-5:])
            
            # Remove outliers using median absolute deviation
            if len(positions) > 3:
                diffs = np.diff(positions)
                median_diff = np.median(diffs)
                mad = np.median(np.abs(diffs - median_diff))
                if mad > 0:
                    outlier_mask = np.abs(diffs - median_diff) < 3 * mad
                    if np.sum(outlier_mask) >= 2:
                        clean_diffs = diffs[outlier_mask]
                        velocity = np.mean(clean_diffs)
                    else:
                        velocity = median_diff
                else:
                    velocity = median_diff
        
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 5:
            self._velocity_history.pop(0)
        
        # Robust velocity smoothing
        if self._velocity_history:
            recent_velocities = np.array(self._velocity_history)
            # Use median for robustness, then smooth
            median_vel = np.median(recent_velocities)
            weights = np.linspace(0.5, 1.0, len(recent_velocities))
            avg_velocity = np.average(recent_velocities, weights=weights)
            # Blend median and weighted average
            final_velocity = 0.3 * median_vel + 0.7 * avg_velocity
        else:
            final_velocity = 0

        # Find paddle position
        paddle_x = self._find_paddle_x(obs)
        self._last_paddle_x = paddle_x

        # Adaptive lookahead with velocity-based scaling
        base_lookahead = 2.5
        velocity_magnitude = abs(final_velocity)
        velocity_factor = min(2.5, velocity_magnitude * 0.6)
        confidence_factor = self._confidence_score ** 0.7  # Slightly less aggressive confidence scaling
        lookahead = base_lookahead + velocity_factor * confidence_factor
        
        self._predicted_ball_x = ball_x + final_velocity * lookahead
        
        # Velocity-adaptive threshold
        base_threshold = 1.8
        speed_threshold = min(2.0, velocity_magnitude * 0.4)
        threshold = base_threshold + speed_threshold * confidence_factor
        
        # Calculate movement with enhanced logic
        target_offset = self._predicted_ball_x - paddle_x
        
        # Enhanced momentum system
        if abs(target_offset) > threshold:
            desired_direction = 1 if target_offset > 0 else -1
            
            # Stronger momentum for consistent direction
            momentum_strength = 0.8 if abs(final_velocity) > 1.5 else 0.6
            self._paddle_momentum = momentum_strength * self._paddle_momentum + (1 - momentum_strength) * desired_direction
            
            # Movement decision with velocity consideration
            move_threshold = 0.25 if abs(final_velocity) > 2.0 else 0.35
            
            if self._paddle_momentum > move_threshold:
                return self.right
            elif self._paddle_momentum < -move_threshold:
                return self.left
            else:
                return self.noop
        else:
            # Near target - fine control
            self._paddle_momentum *= 0.6
            
            # Precision positioning
            if abs(target_offset) > 0.8:
                return self.right if target_offset > 0 else self.left
            else:
                return self.noop

    def _find_ball_x_multiframe(self, obs):
        """Enhanced ball detection using current and previous frames."""
        # Primary detection on current frame
        ball_x = self._find_ball_x_enhanced(obs)
        
        if ball_x is not None:
            return ball_x
            
        # If current frame fails, try multi-frame analysis
        if len(self._frame_buffer) >= 2:
            # Look for movement between frames
            prev_frame = self._frame_buffer[-2]
            current_frame = obs
            
            # Difference-based detection
            diff = np.abs(current_frame.astype(float) - prev_frame.astype(float))
            field_diff = diff[95:185, 8:152, :]
            
            # Look for significant changes (moving ball)
            movement = np.max(field_diff, axis=2) > 30
            coords = np.argwhere(movement)
            
            if len(coords) > 0 and len(coords) < 20:
                # Check if this looks like ball movement
                x_coords = coords[:, 1]
                if len(x_coords) <= 12:  # Small moving object
                    return float(np.mean(x_coords)) + 8
        
        return None

    def _find_ball_x_enhanced(self, obs):
        """Enhanced ball detection using multiple criteria."""
        # Ball region: between bricks and paddle
        field = obs[95:185, 8:152, :]
        
        # Multi-channel bright pixel detection
        r_bright = field[:, :, 0] > 175
        g_bright = field[:, :, 1] > 175
        b_bright = field[:, :, 2] > 175
        
        # Ball is typically bright in multiple channels
        bright_multi = (r_bright.astype(int) + g_bright.astype(int) + b_bright.astype(int)) >= 2
        
        # Very bright pixels (ball center)
        very_bright = np.max(field, axis=2) > 200
        
        # White-ish pixels (common ball color)
        white_like = (field[:, :, 0] > 180) & (field[:, :, 1] > 180) & (field[:, :, 2] > 180)
        
        # Combine criteria
        ball_candidates = bright_multi | very_bright | white_like
        
        coords = np.argwhere(ball_candidates)
        
        if len(coords) == 0:
            return None
            
        # Size filtering - ball should be small
        if len(coords) > 30:
            # Try more restrictive criteria
            strict_candidates = very_bright | white_like
            strict_coords = np.argwhere(strict_candidates)
            if len(strict_coords) > 0 and len(strict_coords) <= 15:
                coords = strict_coords
            else:
                return None
        
        # Compact cluster detection
        if len(coords) <= 10:
            return float(np.mean(coords[:, 1])) + 8
            
        # For larger clusters, find most compact region
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        
        # Use density-based clustering
        center_x = np.median(x_coords)
        center_y = np.median(y_coords)
        
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        close_coords = coords[distances <= 3.5]
        
        if len(close_coords) >= 3:
            return float(np.mean(close_coords[:, 1])) + 8
            
        return None

    def _find_paddle_x(self, obs):
        """Find paddle x-center with fallback."""
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