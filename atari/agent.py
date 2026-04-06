"""
Atari agent — the file the AI researcher modifies.

This is the equivalent of train.py in the LLM autoresearch setup.
Modify anything here: the agent's strategy, observation preprocessing,
internal state, heuristics, learned components, etc.

Usage: python agent.py
"""

import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from prepare import GAME, TIME_BUDGET, make_env, evaluate_agent

# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, input_shape=(4, 84, 84), n_actions=4):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

# ---------------------------------------------------------------------------
# Agent (MODIFY THIS)
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.n_actions = action_space.n
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQN(n_actions=self.n_actions).to(self.device)
        self.target_network = DQN(n_actions=self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.target_update_freq = 1000
        
        # Frame processing
        self.frame_stack = deque(maxlen=4)
        
        # Training state
        self.steps = 0
        self.trained = False
        
    def preprocess_frame(self, frame):
        """Convert RGB frame to grayscale and resize to 84x84"""
        # Convert to grayscale
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        # Resize to 84x84
        from scipy.ndimage import zoom
        resized = zoom(gray, (84/210, 84/160), order=1)
        # Normalize to [0, 1]
        return resized.astype(np.float32) / 255.0
        
    def get_state(self):
        """Stack 4 frames to create state"""
        if len(self.frame_stack) < 4:
            # Pad with zeros if not enough frames
            frames = [np.zeros((84, 84))] * (4 - len(self.frame_stack)) + list(self.frame_stack)
        else:
            frames = list(self.frame_stack)
        return np.array(frames, dtype=np.float32)
    
    def reset(self):
        """Called at the start of each episode."""
        self.frame_stack.clear()
        
    def act(self, obs):
        """Select action using epsilon-greedy policy"""
        # Preprocess and add frame
        processed_frame = self.preprocess_frame(obs)
        self.frame_stack.append(processed_frame)
        
        if not self.trained:
            # During training, use epsilon-greedy
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
        
        # Use network to select action
        state = self.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
            
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# ---------------------------------------------------------------------------
# Train function
# ---------------------------------------------------------------------------

def train():
    """Train the DQN agent within TIME_BUDGET seconds."""
    print(f"Training DQN for {TIME_BUDGET}s")
    
    env = make_env()
    agent = Agent(env.action_space)
    
    start_time = time.time()
    episode = 0
    total_reward = 0
    
    try:
        # Import scipy for image resizing
        import scipy.ndimage
    except ImportError:
        print("Warning: scipy not available, using simple resizing")
        
        # Fallback resize function
        def simple_resize(img, target_shape):
            h, w = img.shape
            th, tw = target_shape
            resized = np.zeros(target_shape)
            for i in range(th):
                for j in range(tw):
                    src_i = int(i * h / th)
                    src_j = int(j * w / tw)
                    resized[i, j] = img[src_i, src_j]
            return resized
        
        # Monkey patch the preprocess function
        def preprocess_frame_simple(self, frame):
            gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            resized = simple_resize(gray, (84, 84))
            return resized.astype(np.float32) / 255.0
        
        agent.preprocess_frame = preprocess_frame_simple.__get__(agent, Agent)
    
    while time.time() - start_time < TIME_BUDGET:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        agent.reset()
        
        episode_reward = 0
        step = 0
        
        while time.time() - start_time < TIME_BUDGET and step < 1000:
            action = agent.act(obs)
            
            result = env.step(action)
            if len(result) == 4:
                next_obs, reward, done, info = result
            else:
                next_obs, reward, done, truncated, info = result
                done = done or truncated
            
            # Store experience
            if len(agent.frame_stack) >= 4:
                state = agent.get_state()
                agent.frame_stack.append(agent.preprocess_frame(next_obs))
                next_state = agent.get_state()
                agent.remember(state, action, reward, next_state, done)
                agent.frame_stack.pop()  # Remove the extra frame we added
            
            obs = next_obs
            episode_reward += reward
            step += 1
            agent.steps += 1
            
            # Train the network
            if agent.steps % 4 == 0:
                agent.replay()
            
            # Update target network
            if agent.steps % agent.target_update_freq == 0:
                agent.update_target_network()
            
            if done:
                break
        
        total_reward += episode_reward
        episode += 1
        
        if episode % 10 == 0:
            avg_reward = total_reward / episode
            elapsed = time.time() - start_time
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Time: {elapsed:.1f}s")
    
    env.close()
    
    # Mark as trained
    agent.trained = True
    
    # Save the trained agent globally
    global trained_agent
    trained_agent = agent
    
    print(f"Training completed. Episodes: {episode}, Total time: {time.time() - start_time:.1f}s")

# Global variable to store trained agent
trained_agent = None

# ---------------------------------------------------------------------------
# Main
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
    
    # Use the trained agent
    if trained_agent is not None:
        agent = trained_agent
        agent.epsilon = 0.0  # No exploration during evaluation
        agent.trained = True
    else:
        # Fallback: create new agent
        env = make_env()
        agent = Agent(env.action_space)
        agent.trained = True
        env.close()

    results = evaluate_agent(agent)
    eval_time = time.time() - t0 - train_time
    total_time = time.time() - t0

    # Print results in parseable format
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