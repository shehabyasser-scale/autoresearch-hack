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
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from prepare import GAME, TIME_BUDGET, make_env, evaluate_agent

# ---------------------------------------------------------------------------
# DQN Network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, action_space_size=4):
        super(DQN, self).__init__()
        # Simple but effective CNN for Atari
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size()
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_space_size)
        
    def _get_conv_out_size(self):
        # Calculate output size after convolutions
        x = torch.zeros(1, 4, 84, 84)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return int(np.prod(x.size()))
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # DQN parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.memory_size = 10000
        self.target_update = 1000
        
        # Initialize networks
        self.q_network = DQN(action_space.n).to(self.device)
        self.target_network = DQN(action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Training state
        self.steps = 0
        self.trained = False
        
        # Frame processing
        self.frame_stack = deque(maxlen=4)
        
    def reset(self):
        """Called at the start of each episode."""
        # Clear frame stack
        self.frame_stack.clear()
        
    def preprocess_frame(self, frame):
        """Convert RGB frame to grayscale and resize to 84x84."""
        # Convert to grayscale
        gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        # Resize to 84x84
        gray = gray[::2, ::2]  # Simple downsampling
        # Crop to roughly square
        if gray.shape[0] > 84:
            start = (gray.shape[0] - 84) // 2
            gray = gray[start:start+84, :]
        if gray.shape[1] > 84:
            start = (gray.shape[1] - 84) // 2
            gray = gray[:, start:start+84]
        
        # Pad if needed
        if gray.shape[0] < 84:
            pad = (84 - gray.shape[0]) // 2
            gray = np.pad(gray, ((pad, 84-gray.shape[0]-pad), (0, 0)), 'constant')
        if gray.shape[1] < 84:
            pad = (84 - gray.shape[1]) // 2
            gray = np.pad(gray, ((0, 0), (pad, 84-gray.shape[1]-pad)), 'constant')
            
        return gray.astype(np.uint8)
    
    def get_state(self, frame):
        """Get state from frame stack."""
        processed = self.preprocess_frame(frame)
        self.frame_stack.append(processed)
        
        # Fill stack if not full
        while len(self.frame_stack) < 4:
            self.frame_stack.append(processed)
            
        # Stack frames
        state = np.stack(self.frame_stack, axis=0)
        return state
    
    def act(self, obs):
        """Select action using epsilon-greedy policy."""
        if not self.trained:
            # Use simple heuristic during training
            return random.randint(0, self.action_space.n - 1)
            
        state = self.get_state(obs)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space.n - 1)
        
        # Get Q-values from network
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(q_values.argmax().cpu().item())
        except Exception as e:
            print(f"Error in act: {e}")
            return random.randint(0, self.action_space.n - 1)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
            
        try:
            # Sample batch
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
            actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
            rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
            next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
            dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
            self.optimizer.step()
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            print(f"Error in replay: {e}")
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

# ---------------------------------------------------------------------------
# Train function
# ---------------------------------------------------------------------------

def train():
    """Train the DQN agent within TIME_BUDGET seconds."""
    print(f"Training DQN agent for {TIME_BUDGET}s")
    
    try:
        env = make_env()
        agent = Agent(env.action_space)
        
        start_time = time.time()
        episode = 0
        total_reward = 0
        
        while time.time() - start_time < TIME_BUDGET - 10:  # Leave some buffer
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            agent.reset()
            episode_reward = 0
            done = False
            step = 0
            
            # Get initial state
            current_state = agent.get_state(state)
            
            while not done and step < 1000:  # Limit steps per episode
                action = random.randint(0, env.action_space.n - 1)  # Random during training
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Process next state
                next_state_processed = agent.get_state(next_state)
                
                # Clip reward
                clipped_reward = np.clip(reward, -1, 1)
                
                # Store experience
                agent.remember(current_state, action, clipped_reward, next_state_processed, done)
                
                current_state = next_state_processed
                episode_reward += reward
                step += 1
                agent.steps += 1
                
                # Train network
                if agent.steps % 4 == 0:
                    agent.replay()
                
                # Update target network
                if agent.steps % agent.target_update == 0:
                    agent.update_target_network()
                
                if time.time() - start_time >= TIME_BUDGET - 10:
                    break
            
            total_reward += episode_reward
            episode += 1
            
            if episode % 10 == 0:
                avg_reward = total_reward / episode
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Steps: {agent.steps}")
        
        agent.trained = True
        agent.epsilon = 0.0  # No exploration during evaluation
        env.close()
        
        print(f"Training completed. Episodes: {episode}, Total steps: {agent.steps}")
        return agent
        
    except Exception as e:
        print(f"Training error: {e}")
        # Return a simple agent that uses random actions
        env = make_env()
        agent = Agent(env.action_space)
        agent.trained = True
        agent.epsilon = 0.0
        env.close()
        return agent

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # Phase 1: Training
    print("=" * 50)
    print("PHASE 1: Training")
    print("=" * 50)
    
    trained_agent = train()
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")
    print()

    # Phase 2: Evaluation
    print("=" * 50)
    print("PHASE 2: Evaluation")
    print("=" * 50)
    
    results = evaluate_agent(trained_agent)
    eval_time = time.time() - t0 - train_time
    total_time = time.time() - t0

    # Print results
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