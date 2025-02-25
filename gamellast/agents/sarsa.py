import numpy as np
import random

class SarsaAgent:
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-4,
                 lambda_trace_on=False, lam=0.9, double_sarsa_on=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.q_table2 = {} if double_sarsa_on else None
        self.lambda_trace_on = lambda_trace_on
        self.lam = lam
        self.eligibility = {}
        self.paused = False
        self.last_loss = 0.0
        self.last_reward = 0.0
        self.callback = None
        self.selected_metric = "Reward"
        self.metric_data = []
        self.training_on = True
        self.double_sarsa_on = double_sarsa_on

    def _to_key(self, state):
        return tuple(state)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        k = self._to_key(state)
        if k not in self.q_table:
            self.q_table[k] = np.zeros(self.action_dim)
            if self.double_sarsa_on and self.q_table2 is not None and k not in self.q_table2:
                self.q_table2[k] = np.zeros(self.action_dim)
        if not self.double_sarsa_on:
            return int(np.argmax(self.q_table[k]))
        else:
            qs = self.q_table[k] + self.q_table2[k]
            return int(np.argmax(qs))

    def _ensure_state(self, st):
        k=self._to_key(st)
        if k not in self.q_table:
            self.q_table[k]=np.zeros(self.action_dim)
        if self.double_sarsa_on and self.q_table2 is not None:
            if k not in self.q_table2:
                self.q_table2[k]=np.zeros(self.action_dim)
        if self.lambda_trace_on and k not in self.eligibility:
            self.eligibility[k]=np.zeros(self.action_dim)
        return k

    def train_step(self, state, action, reward, next_state, next_action, done):
        if self.paused:
            self._callback_post()
            return
        ks=self._ensure_state(state)
        kn=self._ensure_state(next_state)
        q_predict = self.q_table[ks][action]
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * self.q_table[kn][next_action]
        diff=q_target-q_predict
        self.q_table[ks][action]+=self.alpha*diff
        self.last_loss=diff*diff
        self.last_reward=reward
        if self.lambda_trace_on:
            self._update_eligibility(ks,action,diff)
        self._decay_epsilon()
        self._callback_post()

    def train_step_n(self, trajectory, n_steps=3):
        if self.paused:
            self._callback_post()
            return
        for i in range(len(trajectory)):
            if i+n_steps>=len(trajectory):
                break
            state, action, reward=trajectory[i]
            state_n, action_n, _=trajectory[i+n_steps]
            ks=self._ensure_state(state)
            kn=self._ensure_state(state_n)
            g=0.0
            for j in range(i, i+n_steps):
                g+= (self.gamma**(j-i))*trajectory[j][2]
            q_predict=self.q_table[ks][action]
            if (i+n_steps)<len(trajectory)-1:
                an=action_n
                q_target=g + (self.gamma**n_steps)*self.q_table[kn][an]
            else:
                q_target=g
            diff=q_target-q_predict
            self.q_table[ks][action]+=self.alpha*diff
            self.last_loss=diff*diff
            self.last_reward=reward
        self._decay_epsilon()
        self._callback_post()

    def multi_step_train(self, states, actions, rewards, next_states, next_actions, dones):
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
            na=next_actions[i]
            d=dones[i]
            self.train_step(s,a,r,ns,na,d)

    def save(self, path):
        data=(self.q_table, self.epsilon, self.alpha, self.gamma, self.epsilon_min, self.epsilon_decay)
        np.save(path, data, allow_pickle=True)

    def load(self, path):
        data=np.load(path, allow_pickle=True)
        self.q_table=data[0]
        self.epsilon=data[1]
        self.alpha=data[2]
        self.gamma=data[3]
        self.epsilon_min=data[4]
        self.epsilon_decay=data[5]

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

    def _update_eligibility(self, k, a, diff):
        if k not in self.eligibility:
            self.eligibility[k]=np.zeros(self.action_dim)
        self.eligibility[k][a]+=1.0
        for st in list(self.eligibility.keys()):
            self.q_table[st]+= self.alpha*diff*self.eligibility[st]
            self.eligibility[st]= self.gamma*self.lam*self.eligibility[st]
            if np.max(self.eligibility[st])<1e-6:
                del self.eligibility[st]

    def _decay_epsilon(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon-=self.epsilon_decay
            if self.epsilon<self.epsilon_min:
                self.epsilon=self.epsilon_min

    def _callback_post(self):
        if self.callback:
            self.callback({"loss": self.last_loss, "reward": self.last_reward, "epsilon": self.epsilon})

    def select_action_double(self, state):
        if not self.double_sarsa_on:
            return self.select_action(state)
        ks=self._ensure_state(state)
        if random.random()<self.epsilon:
            return random.randint(0, self.action_dim-1)
        qsA=self.q_table[ks]
        qsB=self.q_table2[ks]
        greedyA=int(np.argmax(qsA))
        return greedyA if qsA[greedyA]>=qsB[greedyA] else int(np.argmax(qsB))

    def train_step_double(self, state, action, reward, next_state, next_action, done):
        if self.paused or not self.double_sarsa_on:
            self._callback_post()
            return
        ks=self._ensure_state(state)
        kn=self._ensure_state(next_state)
        qA=self.q_table[ks][action]
        if done:
            target=reward
        else:
            a_star=int(np.argmax(self.q_table[kn]))
            target=reward + self.gamma*self.q_table2[kn][a_star]
        diff=target-qA
        self.q_table[ks][action]+=self.alpha*diff
        self.last_loss=diff*diff
        self.last_reward=reward
        self._decay_epsilon()
        self._callback_post()

    def train_step_double_swap(self, state, action, reward, next_state, next_action, done):
        if self.paused or not self.double_sarsa_on:
            self._callback_post()
            return
        ks=self._ensure_state(state)
        kn=self._ensure_state(next_state)
        qB=self.q_table2[ks][action]
        if done:
            target=reward
        else:
            a_star=int(np.argmax(self.q_table2[kn]))
            target=reward + self.gamma*self.q_table[kn][a_star]
        diff=target-qB
        self.q_table2[ks][action]+=self.alpha*diff
        self.last_loss=diff*diff
        self.last_reward=reward
        self._decay_epsilon()
        self._callback_post()
