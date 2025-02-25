import sys
import argparse
import random
import numpy as np
import os
from environment.environment import RoboCleanerEnv
from agents.qlearning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from ui.pygame_ui import PygameUI
from ui.tkinter_ui import TkinterUI
from ui.pyqt_ui import PyQtUI

class AgentFactory:
    def create_agent(self, algo_name, state_dim, action_dim, **kwargs):
        if algo_name=="qlearning":
            return QLearningAgent(state_dim, action_dim, **kwargs)
        elif algo_name=="sarsa":
            return SarsaAgent(state_dim, action_dim, **kwargs)
        elif algo_name=="dqn":
            return DQNAgent(state_dim, action_dim, **kwargs)
        elif algo_name=="ppo":
            return PPOAgent(state_dim, action_dim, **kwargs)
        else:
            return None

class UISelector:
    def create_ui(self, ui_name):
        if ui_name=="pygame":
            return PygameUI(width=1280, height=960, title="RoboCleaner - PyGame Enhanced", tile_size=48)
        elif ui_name=="tkinter":
            return TkinterUI(width=1200, height=900, title="RoboCleaner", tile_size=40)
        elif ui_name=="pyqt":
            return PyQtUI(width=1200, height=900, title="RoboCleaner", tile_size=40)
        else:
            return PygameUI(width=1280, height=960, title="RoboCleaner - PyGame Enhanced", tile_size=48)

class RunGame:
    def __init__(self):
        self.parser=argparse.ArgumentParser(description="RoboCleaner run_game advanced version")
        self.parser.add_argument("--ui", type=str, default="pygame", help="UI: pygame/tkinter/pyqt")
        self.parser.add_argument("--algo", type=str, default="qlearning", help="Algorithm: qlearning/sarsa/dqn/ppo")
        self.parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
        self.parser.add_argument("--fps", type=int, default=5, help="Frames per second in UI")
        self.parser.add_argument("--random_map", action="store_true", help="Use random environment map")
        self.parser.add_argument("--rows", type=int, default=12, help="Map rows if random_map is used")
        self.parser.add_argument("--cols", type=int, default=12, help="Map cols if random_map is used")
        self.parser.add_argument("--obs_count", type=int, default=8, help="Obstacle count if random_map is used")
        self.parser.add_argument("--dirt_count", type=int, default=8, help="Dirt count if random_map is used")
        self.parser.add_argument("--energy", type=int, default=None, help="Energy limit if any")
        self.parser.add_argument("--time_limit", type=int, default=None, help="Time step limit if any")
        self.parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate if agent supports it")
        self.parser.add_argument("--seed", type=int, default=42, help="Random seed")
        self.parser.add_argument("--dueling", action="store_true", help="Use dueling DQN if algo=dqn")
        self.parser.add_argument("--double_dqn", action="store_true", help="Use double DQN if algo=dqn")
        self.parser.add_argument("--n_step", type=int, default=1, help="n-step for DQN or buffer")
        self.parser.add_argument("--no_prioritized", action="store_true", help="Disable prioritized replay if dqn used")
        self.parser.add_argument("--gae_lambda", type=float, default=0.95, help="PPO lambda for GAE")
        self.parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
        self.parser.add_argument("--k_epochs", type=int, default=10, help="PPO K epoch")
        self.parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
        self.parser.add_argument("--ppo_device", type=str, default="cpu", help="Device for PPO")
        self.args=self.parser.parse_args()
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.env=None
        self.agent=None
        self.ui=None
        self.agent_factory=AgentFactory()
        self.ui_selector=UISelector()

    def setup_environment(self):
        self.env=RoboCleanerEnv(
            use_random=self.args.random_map,
            random_rows=self.args.rows,
            random_cols=self.args.cols,
            random_obstacle_count=self.args.obs_count,
            random_dirt_count=self.args.dirt_count,
            energy_limit=self.args.energy,
            time_limit=self.args.time_limit
        )

    def setup_ui(self):
        self.ui=self.ui_selector.create_ui(self.args.ui)
        self.ui = self.ui_selector.create_ui(self.args.ui)
        self.ui.attach_environment(self.env)

    def setup_agent(self):
        st_dim=self.env.get_state().shape[0]
        act_dim=5
        if self.args.algo=="qlearning":
            self.agent=self.agent_factory.create_agent("qlearning", st_dim, act_dim,
                alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-4)
        elif self.args.algo=="sarsa":
            self.agent=self.agent_factory.create_agent("sarsa", st_dim, act_dim,
                alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=1e-4)
        elif self.args.algo=="dqn":
            self.agent=self.agent_factory.create_agent("dqn", st_dim, act_dim,
                dueling=self.args.dueling,
                double_dqn=self.args.double_dqn,
                lr=self.args.lr,
                capacity=10000,
                batch_size=self.args.batch_size,
                target_update=1000,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=1e-4,
                device="cpu",
                n_step=self.args.n_step
            )
            if not self.args.no_prioritized:
                self.agent.use_prioritized_replay(True)
        elif self.args.algo=="ppo":
            self.agent=self.agent_factory.create_agent("ppo", st_dim, act_dim,
                lr=self.args.lr, gamma=0.99, gae_lambda=self.args.gae_lambda,
                eps_clip=self.args.eps_clip, k_epochs=self.args.k_epochs,
                batch_size=self.args.batch_size, device=self.args.ppo_device
            )
        else:
            sys.exit("Unsupported algo: "+self.args.algo)

    def run_loop(self):
        episodes=self.args.episodes
        fps=self.args.fps
        for ep in range(episodes):
            self.env.level_index=0
            while self.env.level_index<self.env.max_level:
                st=self.env.reset()
                done=False
                if self.args.algo=="sarsa":
                    act=self.agent.select_action(st)
                while not done:
                    self.ui.process_events()
                    l_val=0.0
                    r_val=0.0
                    e_val=0.0
                    if hasattr(self.agent,"last_loss"):
                        l_val=self.agent.last_loss
                    if hasattr(self.agent,"last_reward"):
                        r_val=self.agent.last_reward
                    if hasattr(self.agent,"epsilon"):
                        e_val=self.agent.epsilon
                    self.ui.render(self.env, loss=l_val, reward=r_val, epsilon=e_val)
                    self.ui.set_fps(fps)
                    if self.ui.is_paused():
                        continue
                    if self.args.algo=="sarsa":
                        ns, rew, done, _=self.env.step(act)
                        na=self.agent.select_action(ns)
                        self.agent.train_step(st, act, rew, ns, na, done)
                        st=ns
                        act=na
                    else:
                        chosen=self.agent.select_action(st)
                        ns, rew, done, _=self.env.step(chosen)
                        if self.args.algo=="qlearning":
                            self.agent.train_step(st, chosen, rew, ns, done)
                        elif self.args.algo=="dqn":
                            # 查看 ui 有无 train_loops, 如果 >1 则多次 train_step
                            loops = getattr(self.ui, "train_loops", 1)
                            self.agent.remember(st, chosen, rew, ns, done)
                            for _ in range(loops):
                                self.agent.train_step()
                        elif self.args.algo=="ppo":
                            loops = getattr(self.ui, "train_loops", 1)
                            if done:
                                self.agent.remember(st, chosen, 0.0, rew, True)
                                for _ in range(loops):
                                    self.agent.train_step()
                            else:
                                _, logp_val, val_est = self.agent.select_action(ns)
                                self.agent.remember(st, chosen, logp_val, rew, done, val_est)
                                for _ in range(loops):
                                    self.agent.train_step()
                        st=ns
                self.env.next_level()
        self.ui.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    runner=RunGame()
    runner.setup_environment()
    runner.setup_ui()
    runner.setup_agent()
    runner.run_loop()

if __name__=="__main__":
    main()
