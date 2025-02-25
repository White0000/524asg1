import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-4,
                 double_q=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.q_table2 = {} if double_q else None
        self.double_q = double_q
        self.paused = False
        self.last_loss = 0.0
        self.last_reward = 0.0
        self.callback = None
        self.training_on = True
        self.selected_metric = "Reward"
        self.metric_data = []

    def _to_key(self, state):
        return tuple(state)

    def select_action(self, state):
        if random.random()<self.epsilon:
            return random.randint(0, self.action_dim-1)
        k=self._to_key(state)
        if k not in self.q_table:
            self.q_table[k]=np.zeros(self.action_dim)
            if self.double_q and k not in self.q_table2:
                self.q_table2[k]=np.zeros(self.action_dim)
        if not self.double_q:
            return int(np.argmax(self.q_table[k]))
        else:
            q_sum=self.q_table[k]+self.q_table2[k]
            return int(np.argmax(q_sum))

    def _ensure_state(self, st):
        k=self._to_key(st)
        if k not in self.q_table:
            self.q_table[k]=np.zeros(self.action_dim)
        if self.double_q and k not in self.q_table2:
            self.q_table2[k]=np.zeros(self.action_dim)
        return k

    def train_step(self, state, action, reward, next_state, done):
        if self.paused:
            self._callback_post()
            return
        ks=self._ensure_state(state)
        kn=self._ensure_state(next_state)
        q1_predict=self.q_table[ks][action]
        if done:
            q_target=reward
        else:
            if not self.double_q:
                q_target=reward + self.gamma*np.max(self.q_table[kn])
            else:
                next_a=int(np.argmax(self.q_table[kn]))
                q_target=reward + self.gamma*self.q_table2[kn][next_a]
        diff=q_target-q1_predict
        self.q_table[ks][action]+=self.alpha*diff
        self.last_loss=diff*diff
        self.last_reward=reward
        self._decay_epsilon()
        self._callback_post()

    def train_step_double_swap(self, state, action, reward, next_state, done):
        if self.paused or not self.double_q:
            self._callback_post()
            return
        ks=self._ensure_state(state)
        kn=self._ensure_state(next_state)
        q2_predict=self.q_table2[ks][action]
        if done:
            q_target=reward
        else:
            next_a=int(np.argmax(self.q_table2[kn]))
            q_target=reward + self.gamma*self.q_table[kn][next_a]
        diff=q_target-q2_predict
        self.q_table2[ks][action]+=self.alpha*diff
        self.last_loss=diff*diff
        self.last_reward=reward
        self._decay_epsilon()
        self._callback_post()

    def train_step_n(self, trajectory, n_steps=3):
        if self.paused:
            self._callback_post()
            return
        for i in range(len(trajectory)):
            if i+n_steps>=len(trajectory):
                break
            s, a, r=trajectory[i]
            s_n, a_n, _=trajectory[i+n_steps]
            ks=self._ensure_state(s)
            kn=self._ensure_state(s_n)
            g=0.0
            for j in range(i, i+n_steps):
                g+= (self.gamma**(j-i))*trajectory[j][2]
            if (i+n_steps)<len(trajectory)-1:
                if not self.double_q:
                    q_target=g + (self.gamma**n_steps)*np.max(self.q_table[kn])
                else:
                    next_a=int(np.argmax(self.q_table[kn]))
                    q_target=g + (self.gamma**n_steps)*self.q_table2[kn][next_a]
            else:
                q_target=g
            q_old=self.q_table[ks][a]
            diff=q_target-q_old
            self.q_table[ks][a]+=self.alpha*diff
            self.last_loss=diff*diff
            self.last_reward=r
        self._decay_epsilon()
        self._callback_post()

    def multi_step_train(self, states, actions, rewards, next_states, dones):
        if len(states)!=len(actions):
            return
        if self.paused:
            self._callback_post()
            return
        for i in range(len(states)):
            s=states[i]
            a=actions[i]
            r=rewards[i]
            ns=next_states[i]
            d=dones[i]
            self.train_step(s,a,r,ns,d)

    def save(self, path):
        if not self.double_q:
            data=(self.q_table,self.epsilon,self.alpha,self.gamma,self.epsilon_min,self.epsilon_decay,False)
        else:
            data=(self.q_table,self.q_table2,self.epsilon,self.alpha,self.gamma,self.epsilon_min,self.epsilon_decay,True)
        np.save(path, data, allow_pickle=True)

    def load(self, path):
        data=np.load(path, allow_pickle=True)
        if len(data)==7:
            self.q_table,self.epsilon,self.alpha,self.gamma,self.epsilon_min,self.epsilon_decay,dbq=data
            self.double_q=dbq
            self.q_table2=None
        else:
            self.q_table,self.q_table2,self.epsilon,self.alpha,self.gamma,self.epsilon_min,self.epsilon_decay,self.double_q=data[0:8]

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

    def _decay_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon-=self.epsilon_decay
            if self.epsilon<self.epsilon_min:
                self.epsilon=self.epsilon_min

    def _callback_post(self):
        if self.callback:
            self.callback({"loss": self.last_loss, "reward": self.last_reward, "epsilon": self.epsilon})
