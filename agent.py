import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """
    Simple MLP-based Q-network.
    Input: state (observation) vector
    Output: Q-values for each action (0 ~ 10)
            shape: [batch_size, 11]
    """

    def __init__(self, obs_dim, act_dim=11, hidden_size=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, x):
        return self.net(x)  # shape: (batch, 11)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=1e-5):
        """
        capacity : Maximum size of the replay buffer
        alpha    : Priority exponent (controls how much prioritization is used)
        beta     : Importance sampling exponent (initial value)
        beta_increment : Increment applied to beta at each sampling step
        """
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.epsilon_for_priority = 1e-5

    def push(self, transition):
        """
        transition = (obs, action_idx, reward, next_obs, done)
        """
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            total = self.capacity
        else:
            total = len(self.buffer)

        priorities_array = np.array(self.priorities[:total], dtype=np.float32)
        probs = priorities_array ** self.alpha
        probs /= probs.sum()

        # Gradually increase beta toward 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        indices = np.random.choice(total, batch_size, p=probs)

        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for idx in indices:
            o, a, r, no, d = self.buffer[idx]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            dones.append(d)

        # Importance sampling weights
        # w_i = (N * p_i)^(-beta), normalized by max weight
        p_selected = probs[indices]
        w = (total * p_selected) ** (-self.beta)
        w_max = w.max()
        w_normed = w / w_max  # shape=(batch,)

        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs),
            np.array(dones, dtype=np.float32),
            indices,
            w_normed  # Importance Weight
        )

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            p = abs(err) + self.epsilon_for_priority
            self.priorities[i] = p

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self, 
        obs_dim, 
        act_dim=11, 
        lr=1e-4, 
        gamma=0.99,
        buffer_size=10000, 
        device="cpu",
        n_step=3, # n-step 
        gamma_n=0.99 # gamma for n-step
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.device = device

        self.qnet = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_qnet = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # [!] Prioritized Experience Replay buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.6, beta=0.4, beta_increment=1e-5)

        # Epsilon-greedy exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1e-4

        self.update_target_steps = 1000
        self.learn_step = 0

        self.n_step = n_step
        self.gamma_n = gamma_n 
        self.n_step_buffer = deque(maxlen=n_step)



    def select_action(self, obs):
        if random.random() < self.epsilon:
            # random action (0~10)
            return random.randint(0, self.act_dim - 1)
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.qnet(obs_t)  # shape (1, act_dim)
            action_idx = q_values.argmax(dim=1).item()
            return action_idx

    def store_transition(self, obs, action_idx, reward, next_obs, done):
        # Store transition in temporary n-step buffer
        transition = (obs, action_idx, reward, next_obs, done)
        self.n_step_buffer.append(transition)

        # When n transitions are accumulated, compute n-step return
        if len(self.n_step_buffer) == self.n_step:
            self._push_n_step_transition()

    def _push_n_step_transition(self):
        """
        Compute n-step return:
        G = r0 + gamma*r1 + ... + gamma^(n-1)*r_{n-1}
        """
        # n-step buffer's first transition = (s0,a0,r0,...)
        s0, a0, _, _, _ = self.n_step_buffer[0]

        G = 0.0
        gamma_power = 1.0
        next_s = None
        done_final = False

        for idx, (obs_i, act_i, rew_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
            G += gamma_power * rew_i
            gamma_power *= self.gamma_n
            
            next_s = next_obs_i  # Last transition's next_obs
            if done_i:
                done_final = True
                break

        # next_s = s_{t+n} (or earlier if done)
        self.buffer.push( (s0, a0, G, next_s, done_final) )

        # Delete first transition in n_step_buffer (= left pop)
        # For Safety:
        self.n_step_buffer.popleft()

    def finish_n_step(self):
        """
        At episode termination, flush remaining transitions
        in the n-step buffer with partial returns.
        """
        while len(self.n_step_buffer) > 0:
            self._push_n_step_transition()



    def train_on_batch(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        (
            obs_batch,
            act_batch,
            nstep_return_batch, 
            next_obs_batch,
            done_batch,
            indices,
            is_weights
        ) = self.buffer.sample(batch_size)

        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        act_t = torch.LongTensor(act_batch).view(-1, 1).to(self.device)
        nstep_t = torch.FloatTensor(nstep_return_batch).view(-1, 1).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs_batch).to(self.device)
        done_t = torch.FloatTensor(done_batch).view(-1, 1).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).view(-1, 1).to(self.device)

        q_values = self.qnet(obs_t)  # (B, act_dim)
        q_values_a = q_values.gather(1, act_t)  # (B, 1)

        with torch.no_grad():
            next_q_values = self.target_qnet(next_obs_t)  # (B, act_dim)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]  # (B,1)

            gamma_n = (self.gamma_n ** self.n_step)
            target = nstep_t + (1.0 - done_t) * gamma_n * max_next_q

        td_errors = (target - q_values_a).detach().cpu().numpy().flatten()

        # [!] PER-weighted MSE loss
        # loss = mean( w_i * (TD error)^2 )
        loss = (is_weights_t * (q_values_a - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors)

        # Target network synchronization
        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def save(self, path="dqn_agent.pt"):
        torch.save(
            {
                "qnet": self.qnet.state_dict(),
                "target_qnet": self.target_qnet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path="dqn_agent.pt"):
        data = torch.load(path)
        self.qnet.load_state_dict(data["qnet"])
        self.target_qnet.load_state_dict(data["target_qnet"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon = data["epsilon"]
