import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    """
    간단한 MLP로 구성된 Q네트워크.
    입력: 상태(observation) 벡터
    출력: 각 액션(0~10)에 대한 Q값 (shape: [batch_size, 11])
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
        capacity : 버퍼 최대 용량
        alpha    : 우선순위 분포의 온도 파라미터
        beta     : 중요도 샘플링 가중치 파라미터 (초기값). importance sampling에 쓴다.
        beta_increment: 매번 샘플링할 때마다 beta를 조금씩 올리는 정도
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

        # beta를 조금씩 증가
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

        # [추가] importance sampling weight 계산
        # w_i = ( N * p_i )^-beta  (또는 1/(N* p_i) ^ beta ) -> 정규화 필요
        # p_i = probs[idx]
        # N = total
        # 가장 큰 w_i로 나누어 정규화(관례적으로)
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
            w_normed  # 중요도 가중치
        )

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            p = abs(err) + self.epsilon_for_priority
            self.priorities[i] = p

    def __len__(self):
        return len(self.buffer)



# (3) DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(
        self, 
        obs_dim, 
        act_dim=11, 
        lr=1e-4, 
        gamma=0.99,
        buffer_size=10000, 
        device="cpu",
        n_step=3, # n-step 값. 원하는 대로 3, 5 등으로 둘 수 있다.
        gamma_n=0.99 # 혹은 n-step 용 감마. 보통 동일 gamma 써도 됨.
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.device = device

        self.qnet = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_qnet = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        # [중요] 우선순위 버퍼에 beta를 포함
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.6, beta=0.4, beta_increment=1e-5)

        # Epsilon-Greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1e-4

        self.update_target_steps = 1000
        self.learn_step = 0

        self.n_step = n_step # 추가
        self.gamma_n = gamma_n # 추가
        self.n_step_buffer = deque(maxlen=n_step) # 추가



    def select_action(self, obs):
        if random.random() < self.epsilon:
            # 0~10 사이 랜덤 정수
            return random.randint(0, self.act_dim - 1)
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.qnet(obs_t)  # shape (1, act_dim)
            action_idx = q_values.argmax(dim=1).item()
            return action_idx

    def store_transition(self, obs, action_idx, reward, next_obs, done):
        # 1) 우선 n-step 임시 버퍼에 저장
        transition = (obs, action_idx, reward, next_obs, done)
        self.n_step_buffer.append(transition)

        # 2) 버퍼에 n개 쌓이면 -> n-step return 계산해서 최종 저장
        if len(self.n_step_buffer) == self.n_step:
            self._push_n_step_transition()
        
        # 만약 done=True면, 남은 transition도 처리해야 하지만,
        # done은 step()안에서 처리 -> main 루프 끝나면 partial n-step 처리
        # -> 이 처리는 train_on_batch 또는 episode 끝에서 해도 됨


    def _push_n_step_transition(self):
        """
        n_step_buffer 내 첫 번째 transition부터 n개를 꺼내
        G = r0 + gamma*r1 + ... + gamma^(n-1)*rn-1
        next_state = s_{t+n}, done = any done in the next steps? or the last one
        """
        # n-step 버퍼의 첫 transition = (s0,a0,r0,...)
        s0, a0, _, _, _ = self.n_step_buffer[0]

        # 누적 보상과 마지막 next_obs, done을 구하기
        G = 0.0
        gamma_power = 1.0
        next_s = None
        done_final = False

        for idx, (obs_i, act_i, rew_i, next_obs_i, done_i) in enumerate(self.n_step_buffer):
            G += gamma_power * rew_i
            gamma_power *= self.gamma_n
            
            next_s = next_obs_i  # 마지막 transition의 next_obs
            if done_i:
                # n-step 도중 done 발생 -> 그 시점 이후는 reward가 없으므로 break
                done_final = True
                break

        # 이 시점에서 G = n-step 혹은 중간 done 시까지의 누적 보상
        # next_s = s_{t+n} (or earlier if done)
        # done_final = True if 중간에 끝났다면

        self.buffer.push( (s0, a0, G, next_s, done_final) )

        # n_step_buffer에서 첫 transition 제거(= left pop)
        # 하지만 deque(maxlen=n_step)이 자동으로 밀어내긴 함
        # 안전하게 해주려면:
        self.n_step_buffer.popleft()

    def finish_n_step(self):
        """
        에피소드 종료 시, n_step_buffer에 남은 transition들을
        partial n-step return 형태로 계산해 push한다
        """
        while len(self.n_step_buffer) > 0:
            self._push_n_step_transition()



    def train_on_batch(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        (
            obs_batch,
            act_batch,
            nstep_return_batch, # <- 여기서 rew_batch 대신 nstep_return_batch 가 온다.
            next_obs_batch,
            done_batch,
            indices,
            is_weights  # 중요도 가중치
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

        # [중요] PER: 중요도 가중치 * TD error^2 의 평균
        # MSE 대신 가중치된 MSE를 쓴다
        # loss = mean( w_i * (TD error)^2 )
        loss = (is_weights_t * (q_values_a - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER 우선순위 갱신
        self.buffer.update_priorities(indices, td_errors)

        # 타깃 Q넷 동기화
        self.learn_step += 1
        if self.learn_step % self.update_target_steps == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        # epsilon decay
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
