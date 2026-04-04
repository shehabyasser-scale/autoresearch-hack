"""
Atari DQN agent -- the file the AI researcher modifies.
Usage: python agent.py
"""

import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from prepare import GAME, TIME_BUDGET, make_env, evaluate_agent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs):
    gray = np.mean(obs[32:195, :, :], axis=2).astype(np.float32)
    gray = gray[::2, ::2]
    result = np.zeros((84, 84), dtype=np.float32)
    rh, rw = min(gray.shape[0], 84), min(gray.shape[1], 84)
    result[:rh, :rw] = gray[:rh, :rw]
    return result / 255.0

class DQN(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
        )
        dummy = torch.zeros(1, 4, 84, 84)
        conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_out, 256), nn.ReLU(), nn.Linear(256, num_actions))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(ns), np.array(d, dtype=np.float32)
    def __len__(self):
        return len(self.buffer)

BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50000
TARGET_UPDATE_FREQ = 1000
REPLAY_MIN_SIZE = 1000
REPLAY_CAPACITY = 50000

policy_net = None
target_net = None

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.frame_stack = deque(maxlen=4)
    def reset(self):
        self.frame_stack.clear()
    def _get_state(self, obs):
        frame = preprocess(obs)
        while len(self.frame_stack) < 4:
            self.frame_stack.append(frame)
        self.frame_stack.append(frame)
        return np.array(self.frame_stack, dtype=np.float32)
    def act(self, obs):
        state = self._get_state(obs)
        if policy_net is None:
            return self.action_space.sample()
        with torch.no_grad():
            return policy_net(torch.from_numpy(state).unsqueeze(0).to(DEVICE)).argmax(1).item()

def train():
    global policy_net, target_net
    print("Training DQN for up to " + str(TIME_BUDGET) + "s on " + str(DEVICE) + "...")
    env = make_env()
    num_actions = env.action_space.n
    policy_net = DQN(num_actions).to(DEVICE)
    target_net = DQN(num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY)
    obs, info = env.reset()
    fstack = deque(maxlen=4)
    fr = preprocess(obs)
    for _ in range(4): fstack.append(fr)
    total_steps = 0; ep_count = 0; ep_reward = 0.0; tot_reward = 0.0
    losses = []; epsilon = EPSILON_START; t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET: break
        epsilon = max(EPSILON_END, EPSILON_START - total_steps / EPSILON_DECAY_STEPS)
        state = np.array(fstack, dtype=np.float32)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(torch.from_numpy(state).unsqueeze(0).to(DEVICE)).argmax(1).item()
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        nf = preprocess(next_obs); fstack.append(nf)
        next_state = np.array(fstack, dtype=np.float32)
        replay.push(state, action, reward, next_state, done)
        ep_reward += reward; total_steps += 1
        if done:
            obs, info = env.reset(); fr = preprocess(obs); fstack.clear()
            for _ in range(4): fstack.append(fr)
            tot_reward += ep_reward; ep_count += 1; ep_reward = 0.0
        else:
            obs = next_obs
        if len(replay) >= REPLAY_MIN_SIZE:
            sb, ab, rb, nsb, db = replay.sample(BATCH_SIZE)
            st = torch.from_numpy(sb).to(DEVICE)
            at = torch.from_numpy(ab).long().to(DEVICE)
            rt = torch.from_numpy(rb).to(DEVICE)
            nst = torch.from_numpy(nsb).to(DEVICE)
            dt = torch.from_numpy(db).to(DEVICE)
            qv = policy_net(st).gather(1, at.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = rt + GAMMA * target_net(nst).max(1)[0] * (1 - dt)
            loss = nn.functional.smooth_l1_loss(qv, tgt)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step(); losses.append(loss.item())
        if total_steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if total_steps % 5000 == 0:
            al = np.mean(losses[-100:]) if losses else 0.0
            ar = tot_reward / max(ep_count, 1)
            print("  step=" + str(total_steps) + " | eps=" + str(round(epsilon, 3)) + " | loss=" + str(round(al, 4)) + " | episodes=" + str(ep_count) + " | avg_reward=" + str(round(ar, 2)) + " | time=" + str(int(elapsed)) + "s/" + str(TIME_BUDGET) + "s")
    env.close()
    tt = time.time() - t0; ar = tot_reward / max(ep_count, 1)
    print("Training complete: steps=" + str(total_steps) + " episodes=" + str(ep_count) + " avg_reward=" + str(round(ar, 2)) + " epsilon=" + str(round(epsilon, 3)) + " time=" + str(round(tt, 1)) + "s")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 50); print("PHASE 1: Training"); print("=" * 50)
    train()
    train_time = time.time() - t0
    print("Training time: " + str(round(train_time, 1)) + "s"); print()
    print("=" * 50); print("PHASE 2: Evaluation"); print("=" * 50)
    env = make_env(); agent = Agent(env.action_space); env.close()
    results = evaluate_agent(agent)
    eval_time = time.time() - t0 - train_time
    total_time = time.time() - t0
    print(); print("---")
    for k in ["mean_reward", "std_reward", "min_reward", "max_reward"]:
        print(k + ":" + " " * (18 - len(k)) + str(round(results[k], 4)))
    print("mean_steps:       " + str(round(results["mean_steps"], 1)))
    print("total_steps:      " + str(results["total_steps"]))
    print("episodes:         " + str(results["episodes"]))
    print("training_seconds: " + str(round(train_time, 1)))
    print("eval_seconds:     " + str(round(eval_time, 1)))
    print("total_seconds:    " + str(round(total_time, 1)))
