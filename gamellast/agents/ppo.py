import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor_head = nn.Linear(256, action_dim)
        self.critic_head = nn.Linear(256, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_logits = self.actor_head(x)
        critic_value = self.critic_head(x)
        return actor_logits, critic_value

class RolloutBuffer:
    def __init__(self):
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.values=[]
    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    def multi_store(self, state_list, action_list, logp_list, reward_list, done_list, value_list):
        for i in range(len(state_list)):
            self.store(state_list[i], action_list[i], logp_list[i], reward_list[i], done_list[i], value_list[i])
    def clear(self):
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]
        self.dones=[]
        self.values=[]

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 eps_clip=0.2, k_epochs=10, batch_size=64, device="cpu",
                 adaptive_kl=False, max_batch_size=None):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.lr=lr
        self.gamma=gamma
        self.gae_lambda=gae_lambda
        self.eps_clip=eps_clip
        self.k_epochs=k_epochs
        self.batch_size=batch_size
        self.device=torch.device(device)
        self.adaptive_kl=adaptive_kl
        self.max_batch_size=max_batch_size if max_batch_size else batch_size
        self.buffer=RolloutBuffer()
        self.model=ActorCritic(state_dim,action_dim).to(self.device)
        self.optimizer=optim.Adam(self.model.parameters(),lr=self.lr)
        self.step_count=0
        self.paused=False
        self.last_loss=0.0
        self.last_reward=0.0
        self.callback=None
        self.selected_metric="Reward"
        self.metric_data=[]
        self.metric_data2=[]
        self.kl_coeff=0.2
        self.kl_target=0.01
    def select_action(self, state):
        st=torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            actor_logits, val=self.model(st)
        dist=torch.distributions.Categorical(logits=actor_logits)
        action=dist.sample()
        logp=dist.log_prob(action)
        return action.item(), logp.item(), val.item()
    def remember(self, state, action, logp, reward, done, value):
        self.last_reward=reward
        self.buffer.store(state,action,logp,reward,done,value)
    def multi_remember(self, states, actions, logps, rewards, dones, values):
        self.buffer.multi_store(states,actions,logps,rewards,dones,values)
    def train_step(self):
        if self.paused:
            self._callback()
            return
        if not self.buffer.states:
            self._callback()
            return
        rewards=[]
        values=[]
        dones=[]
        for r,v,d in zip(self.buffer.rewards,self.buffer.values,self.buffer.dones):
            rewards.append(r)
            values.append(v)
            dones.append(d)
        advantages=[]
        gae=0.0
        values.append(values[-1])
        for i in reversed(range(len(rewards))):
            delta=rewards[i]+self.gamma*values[i+1]*(1-dones[i])-values[i]
            gae=delta+self.gamma*self.gae_lambda*(1-dones[i])*gae
            advantages.insert(0,gae)
        adv_t=torch.FloatTensor(advantages).to(self.device)
        returns=adv_t+torch.FloatTensor(values[:-1]).to(self.device)
        st=torch.FloatTensor(np.array(self.buffer.states,dtype=np.float32)).to(self.device)
        ac=torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp=torch.FloatTensor(self.buffer.log_probs).to(self.device)
        self.buffer.clear()
        for _ in range(self.k_epochs):
            idxs=np.arange(st.size(0))
            np.random.shuffle(idxs)
            start=0
            while start<st.size(0):
                end=min(start+self.batch_size, st.size(0))
                if end-start>self.max_batch_size:
                    end=start+self.max_batch_size
                bidx=idxs[start:end]
                b_st=st[bidx]
                b_ac=ac[bidx]
                b_old_lp=old_lp[bidx]
                b_returns=returns[bidx]
                b_adv=adv_t[bidx]
                logits,vals=self.model(b_st)
                dist=torch.distributions.Categorical(logits=logits)
                new_lp=dist.log_prob(b_ac)
                ratio=torch.exp(new_lp-b_old_lp)
                surr1=ratio*b_adv
                surr2=torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)*b_adv
                actor_loss=-torch.min(surr1,surr2).mean()
                critic_loss=F.mse_loss(vals.squeeze(), b_returns)
                ent=dist.entropy().mean()
                total_loss=actor_loss+0.5*critic_loss-0.01*ent
                if self.adaptive_kl:
                    kl=(b_old_lp-new_lp).mean().abs().item()
                    if kl>self.kl_target*2:
                        self.kl_coeff*=1.5
                    elif kl<self.kl_target/2:
                        self.kl_coeff*=0.5
                    total_loss+= self.kl_coeff*kl
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.last_loss=total_loss.item()
                start=end
        self._callback()
    def multi_train_step(self, steps=1):
        for _ in range(steps):
            self.train_step()
    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kl_coeff": self.kl_coeff
        }, path)
    def load(self, path):
        data=torch.load(path,map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        if "kl_coeff" in data:
            self.kl_coeff=data["kl_coeff"]
    def pause_training(self):
        self.paused=True
    def resume_training(self):
        self.paused=False
    def register_callback(self, func):
        self.callback=func
    def set_training_param(self, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self,k):
                setattr(self,k,v)
    def adjust_learning_rate(self, new_lr):
        for g in self.optimizer.param_groups:
            g["lr"]=new_lr
        self.lr=new_lr
    def _callback(self):
        if self.callback:
            self.callback({"loss": self.last_loss, "reward": self.last_reward})
    def debug_set_metric_data(self, val1, val2=0.0):
        self.metric_data.append(val1)
        self.metric_data2.append(val2)
    def get_model_parameters(self):
        return list(self.model.parameters())
