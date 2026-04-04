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
    # Physics-aware positioning with adaptive timing and wall bounce prediction
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
        self._ball_y_history = []
        self._wall_bounce_prediction = None
        self._reaction_delay = 0
        self._game_phase = "early"  # early, mid, late

    def act(self, obs):
        """
        Given a (210, 160, 3) uint8 RGB observation, return an action.

        Physics-aware strategy with wall bounce prediction and adaptive timing.
        """
        self._steps += 1

        # Determine game phase
        if self._steps < 100:
            self._game_phase = "early"
        elif self._steps < 500:
            self._game_phase = "mid"
        else:
            self._game_phase = "late"

        # Fire to launch the ball at the start
        if not self._fired:
            self._fired = True
            return self.fire

        # Store frame for multi-frame analysis
        self._frame_buffer.append(obs.copy())
        if len(self._frame_buffer) > 4:
            self._frame_buffer.pop(0)

        # Find ball position and y-coordinate
        ball_pos = self._find_ball_position_enhanced(obs)
        
        if ball_pos is None:
            self._no_ball_count += 1
            self._confidence_score = max(0, self._confidence_score - 0.3)
            
            # Use physics prediction if available
            if self._wall_bounce_prediction is not None and self._confidence_score > 0.2:
                ball_x = self._wall_bounce_prediction
                ball_x = max(8, min(152, ball_x))
            elif len(self._ball_history) >= 3 and self._predicted_ball_x is not None:
                ball_x = self._predicted_ball_x
                ball_x = max(8, min(152, ball_x))
            else:
                ball_x = None
            
            if ball_x is None:
                # Emergency recovery
                if self._no_ball_count > 20:
                    self._emergency_mode = True
                    return self.fire
                
                # Center positioning during ball loss
                paddle_x = self._find_paddle_x(obs)
                center_x = 80
                if abs(paddle_x - center_x) > 25:
                    return self.right if center_x > paddle_x else self.left
                return self.noop
        else:
            ball_x, ball_y = ball_pos
            self._no_ball_count = 0
            self._confidence_score = min(1.0, self._confidence_score + 0.6)
            self._last_known_ball_x = ball_x
            self._emergency_mode = False

        # Update ball history with both x and y
        self._ball_history.append(ball_x)
        if ball_pos is not None:
            self._ball_y_history.append(ball_pos[1])
        if len(self._ball_history) > 10:
            self._ball_history.pop(0)
        if len(self._ball_y_history) > 10:
            self._ball_y_history.pop(0)

        # Enhanced velocity computation with physics awareness
        velocity_x = 0
        velocity_y = 0
        
        if len(self._ball_history) >= 4:
            # X velocity with outlier filtering
            positions_x = np.array(self._ball_history[-6:])
            diffs_x = np.diff(positions_x)
            if len(diffs_x) >= 3:
                median_diff = np.median(diffs_x)
                mad = np.median(np.abs(diffs_x - median_diff))
                if mad > 0:
                    outlier_mask = np.abs(diffs_x - median_diff) < 2.5 * mad
                    if np.sum(outlier_mask) >= 2:
                        velocity_x = np.mean(diffs_x[outlier_mask])
                    else:
                        velocity_x = median_diff
                else:
                    velocity_x = median_diff
        
        if len(self._ball_y_history) >= 4:
            # Y velocity for physics prediction
            positions_y = np.array(self._ball_y_history[-6:])
            diffs_y = np.diff(positions_y)
            if len(diffs_y) >= 2:
                velocity_y = np.mean(diffs_y[-3:])
        
        self._velocity_history.append(velocity_x)
        if len(self._velocity_history) > 6:
            self._velocity_history.pop(0)

        # Physics-based wall bounce prediction
        self._wall_bounce_prediction = self._predict_wall_bounce(ball_x, velocity_x)

        # Find paddle position
        paddle_x = self._find_paddle_x(obs)
        self._last_paddle_x = paddle_x

        # Adaptive reaction timing based on game phase and ball speed
        ball_speed = abs(velocity_x)
        if self._game_phase == "early":
            base_lookahead = 3.0
            reaction_threshold = 1.5
        elif self._game_phase == "mid":
            base_lookahead = 2.5
            reaction_threshold = 1.8
        else:  # late game - more aggressive
            base_lookahead = 2.0
            reaction_threshold = 2.2

        # Enhanced lookahead with physics consideration
        speed_factor = min(3.0, ball_speed * 0.8)
        confidence_factor = self._confidence_score ** 0.5
        
        # Use wall bounce prediction if available and confident
        if self._wall_bounce_prediction is not None and self._confidence_score > 0.4:
            target_x = self._wall_bounce_prediction
        else:
            lookahead = base_lookahead + speed_factor * confidence_factor
            target_x = ball_x + velocity_x * lookahead

        # Adaptive positioning strategy
        target_offset = target_x - paddle_x
        
        # Dynamic threshold based on ball speed and Y position
        speed_threshold = min(2.5, ball_speed * 0.5)
        threshold = reaction_threshold + speed_threshold * confidence_factor
        
        # Add Y-position based urgency
        if ball_pos is not None and ball_pos[1] > 70:  # Ball getting close to paddle
            threshold *= 0.7  # React faster when ball is close
            
        # Enhanced momentum system with physics awareness
        if abs(target_offset) > threshold:
            desired_direction = 1 if target_offset > 0 else -1
            
            # Stronger momentum for high-speed balls
            momentum_strength = 0.9 if ball_speed > 2.0 else 0.7
            self._paddle_momentum = momentum_strength * self._paddle_momentum + (1 - momentum_strength) * desired_direction
            
            # Movement decision with speed-adaptive threshold
            move_threshold = 0.2 if ball_speed > 2.5 else 0.3
            
            if self._paddle_momentum > move_threshold:
                return self.right
            elif self._paddle_momentum < -move_threshold:
                return self.left
            else:
                return self.noop
        else:
            # Fine positioning near target
            self._paddle_momentum *= 0.5
            
            # Precision control
            if abs(target_offset) > 1.0:
                return self.right if target_offset > 0 else self.left
            else:
                return self.noop

    def _predict_wall_bounce(self, ball_x, velocity_x):
        """Predict ball position after wall bounces."""
        if abs(velocity_x) < 0.5:
            return None
            
        # Simulate ball movement with wall bounces
        sim_x = ball_x
        sim_vx = velocity_x
        steps_ahead = 15
        
        for _ in range(steps_ahead):
            sim_x += sim_vx
            
            # Check for wall collision
            if sim_x <= 8:  # Left wall
                sim_x = 8 + (8 - sim_x)
                sim_vx = -sim_vx * 0.95  # Slight energy loss
            elif sim_x >= 152:  # Right wall
                sim_x = 152 - (sim_x - 152)
                sim_vx = -sim_vx * 0.95
                
        return sim_x

    def _find_ball_position_enhanced(self, obs):
        """Enhanced ball detection returning both x and y coordinates."""
        # Ball region: between bricks and paddle
        field = obs[95:185, 8:152, :]
        
        # Multi-criteria detection
        r_bright = field[:, :, 0] > 170
        g_bright = field[:, :, 1] > 170
        b_bright = field[:, :, 2] > 170
        
        # Ball is typically bright in multiple channels
        bright_multi = (r_bright.astype(int) + g_bright.astype(int) + b_bright.astype(int)) >= 2
        
        # Very bright pixels
        very_bright = np.max(field, axis=2) > 195
        
        # White-like pixels
        white_like = (field[:, :, 0] > 175) & (field[:, :, 1] > 175) & (field[:, :, 2] > 175)
        
        # Combine criteria
        ball_candidates = bright_multi | very_bright | white_like
        
        coords = np.argwhere(ball_candidates)
        
        if len(coords) == 0:
            return self._find_ball_multiframe(obs)
            
        # Size filtering
        if len(coords) > 25:
            strict_candidates = very_bright | white_like
            strict_coords = np.argwhere(strict_candidates)
            if len(strict_coords) > 0 and len(strict_coords) <= 12:
                coords = strict_coords
            else:
                return None
        
        # Return center of mass
        if len(coords) <= 12:
            y_pos = float(np.mean(coords[:, 0]))
            x_pos = float(np.mean(coords[:, 1])) + 8
            return (x_pos, y_pos)
            
        return None

    def _find_ball_multiframe(self, obs):
        """Multi-frame ball detection fallback."""
        if len(self._frame_buffer) >= 2:
            prev_frame = self._frame_buffer[-2]
            current_frame = obs
            
            diff = np.abs(current_frame.astype(float) - prev_frame.astype(float))
            field_diff = diff[95:185, 8:152, :]
            
            movement = np.max(field_diff, axis=2) > 25
            coords = np.argwhere(movement)
            
            if len(coords) > 0 and len(coords) < 15:
                y_pos = float(np.mean(coords[:, 0]))
                x_pos = float(np.mean(coords[:, 1])) + 8
                return (x_pos, y_pos)
        
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