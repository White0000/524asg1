import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingDQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.v_fc = nn.Linear(256, 1)
        self.a_fc = nn.Linear(256, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        v = self.v_fc(x)
        a = self.a_fc(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

class ReplayBuffer:
    def __init__(self, capacity, n_step=1, gamma=0.99):
        self.buffer = collections.deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.nstep_queue = collections.deque()
    def _discounted_reward(self, transitions):
        r=0.0
        for i,(st,ac,re,ns,done) in enumerate(transitions):
            r+= (self.gamma**i)*re
        return r
    def push(self, state, action, reward, next_state, done):
        self.nstep_queue.append((state, action, reward, next_state, done))
        if len(self.nstep_queue)<self.n_step:
            return
        R=self._discounted_reward(self.nstep_queue)
        s0,a0,_,_,_=self.nstep_queue[0]
        _,_,_,ns,dn=self.nstep_queue[-1]
        self.buffer.append((s0,a0,R,ns,dn))
        if not done:
            pass
        self.nstep_queue.popleft()
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states,dtype=np.float32), np.array(actions), np.array(rewards,dtype=np.float32), np.array(next_states,dtype=np.float32), np.array(dones,dtype=np.float32)
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=1, gamma=0.99):
        self.capacity=capacity
        self.alpha=alpha
        self.buffer=[]
        self.pos=0
        self.priorities=np.zeros((capacity,),dtype=np.float32)
        self.n_step=n_step
        self.gamma=gamma
        self.nstep_queue=collections.deque()
    def _discounted_reward(self, transitions):
        r=0.0
        for i,(st,ac,re,ns,dn) in enumerate(transitions):
            r+= (self.gamma**i)*re
        return r
    def push(self, state, action, reward, next_state, done):
        self.nstep_queue.append((state,action,reward,next_state,done))
        if len(self.nstep_queue)<self.n_step:
            return
        R=self._discounted_reward(self.nstep_queue)
        s0,a0,_,_,_=self.nstep_queue[0]
        _,_,_,ns,dn=self.nstep_queue[-1]
        max_prio=self.priorities.max() if self.buffer else 1.0
        data=(s0,a0,R,ns,dn)
        if len(self.buffer)<self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos]=data
        self.priorities[self.pos]=max_prio
        self.pos=(self.pos+1)%self.capacity
        self.nstep_queue.popleft()
    def sample(self,batch_size,beta=0.4):
        if len(self.buffer)==self.capacity:
            prios=self.priorities
        else:
            prios=self.priorities[:len(self.buffer)]
        probs=prios**self.alpha
        probs/=probs.sum()
        indices=np.random.choice(len(self.buffer),batch_size,p=probs)
        batch=[self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones= zip(*batch)
        states=np.array(states,dtype=np.float32)
        actions=np.array(actions)
        rewards=np.array(rewards,dtype=np.float32)
        next_states=np.array(next_states,dtype=np.float32)
        dones=np.array(dones,dtype=np.float32)
        total=len(self.buffer)
        weights=(total*probs[indices])**(-beta)
        weights/=weights.max()
        return states,actions,rewards,next_states,dones,indices,weights
    def update_priorities(self,batch_indices,batch_priorities):
        for idx,prio in zip(batch_indices,batch_priorities):
            self.priorities[idx]=prio
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, dueling=False, double_dqn=False,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=1e-4, capacity=10000, batch_size=64,
                 target_update=1000, device="cpu", n_step=1):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.dueling=dueling
        self.double_dqn=double_dqn
        self.lr=lr
        self.gamma=gamma
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.capacity=capacity
        self.batch_size=batch_size
        self.target_update=target_update
        self.device=torch.device(device)
        self.n_step=n_step
        self.policy_net=None
        self.target_net=None
        self.optimizer=None
        self.memory=None
        self.use_prioritized=False
        self.alpha=0.6
        self.beta=0.4
        self.beta_increment=1e-5
        self.update_count=0
        self.last_loss=0.0
        self.last_reward=0.0
        self._build_model()

    def _build_model(self):
        if self.dueling:
            self.policy_net=DuelingDQNNetwork(self.state_dim,self.action_dim).to(self.device)
            self.target_net=DuelingDQNNetwork(self.state_dim,self.action_dim).to(self.device)
        else:
            self.policy_net=DQNNetwork(self.state_dim,self.action_dim).to(self.device)
            self.target_net=DQNNetwork(self.state_dim,self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=self.lr)
        self.memory=ReplayBuffer(self.capacity,n_step=self.n_step,gamma=self.gamma)

    def use_prioritized_replay(self,use_it):
        self.use_prioritized=use_it
        if use_it:
            self.memory=PrioritizedReplayBuffer(self.capacity,alpha=self.alpha,n_step=self.n_step,gamma=self.gamma)

    def select_action(self,state):
        if np.random.rand()<self.epsilon:
            return np.random.randint(self.action_dim)
        st=torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            qv=self.policy_net(st)
        return int(qv.argmax(dim=1).item())

    def remember(self,state,action,reward,next_state,done):
        self.last_reward=reward
        self.memory.push(state,action,reward,next_state,done)

    def multi_remember(self,states,actions,rewards,next_states,dones):
        for i in range(len(states)):
            self.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def train_step(self):
        if len(self.memory)<self.batch_size:
            return
        if not self.use_prioritized:
            states,actions,rewards,next_states,dones=self.memory.sample(self.batch_size)
            weights=np.ones_like(rewards,dtype=np.float32)
            indices=None
        else:
            states,actions,rewards,next_states,dones,indices,weights=self.memory.sample(self.batch_size,beta=self.beta)
            self.beta=min(1.0,self.beta+self.beta_increment)
        st=torch.FloatTensor(states).to(self.device)
        ac=torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rw=torch.FloatTensor(rewards).to(self.device)
        ns=torch.FloatTensor(next_states).to(self.device)
        dn=torch.FloatTensor(dones).to(self.device)
        ws=torch.FloatTensor(weights).to(self.device)
        q_values=self.policy_net(st).gather(1, ac).squeeze(1)
        with torch.no_grad():
            if not self.double_dqn:
                max_next_q=self.target_net(ns).max(dim=1)[0]
            else:
                next_act=self.policy_net(ns).argmax(dim=1).unsqueeze(1)
                max_next_q=self.target_net(ns).gather(1, next_act).squeeze(1)
        targets=rw+self.gamma*max_next_q*(1-dn)
        diff=q_values - targets
        loss=(diff.pow(2)*ws).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss=loss.item()
        if self.use_prioritized:
            prios=abs(diff.detach().cpu().numpy())+1e-6
            if indices is not None:
                self.memory.update_priorities(indices,prios)
        self.update_count+=1
        self._decay_epsilon()
        if self.update_count%self.target_update==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def multi_train_step(self,steps=1):
        for _ in range(steps):
            self.train_step()

    def save(self,path):
        torch.save({
            "policy_net":self.policy_net.state_dict(),
            "target_net":self.target_net.state_dict(),
            "epsilon":self.epsilon,
            "double_dqn":self.double_dqn
        },path)

    def load(self,path):
        checkpoint=torch.load(path,map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon=checkpoint["epsilon"]
        self.double_dqn=checkpoint["double_dqn"]

    def set_learning_rate(self,new_lr):
        for g in self.optimizer.param_groups:
            g["lr"]=new_lr
        self.lr=new_lr

    def _decay_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon-=self.epsilon_decay
            if self.epsilon<self.epsilon_min:
                self.epsilon=self.epsilon_min

    def close(self):
        pass

    def _callback_post(self):
        pass
