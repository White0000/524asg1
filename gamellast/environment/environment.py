import numpy as np
import random
from levels.levels import LEVELS
from utils.utils import random_obstacles, random_dirts

class RoboCleanerEnv:
    def __init__(self, use_random=False, random_rows=8, random_cols=8,
                 random_obstacle_count=5, random_dirt_count=5,
                 energy_limit=None, time_limit=None, n_agents=1):
        self.use_random = use_random
        self.random_rows = random_rows
        self.random_cols = random_cols
        self.random_obstacle_count = random_obstacle_count
        self.random_dirt_count = random_dirt_count
        self.energy_limit = energy_limit
        self.time_limit = time_limit
        self.n_agents = n_agents
        self.level_index = 0
        self.max_level = len(LEVELS) if not use_random else 1
        self.rows = 0
        self.cols = 0
        self.obstacles = set()
        self.dirts = set()
        self.agent_positions = []
        self.done_flags = [False]*n_agents
        self.done = False
        self.score = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.energy = [None]*n_agents
        self.load_level(self.level_index)

    def load_level(self, idx):
        if self.use_random:
            self.rows = self.random_rows
            self.cols = self.random_cols
            self.obstacles = random_obstacles(self.rows, self.cols, self.random_obstacle_count)
            self.dirts = random_dirts(self.rows, self.cols, self.random_dirt_count, self.obstacles)
            self.agent_positions.clear()
            for _ in range(self.n_agents):
                while True:
                    rr = random.randint(0, self.rows - 1)
                    cc = random.randint(0, self.cols - 1)
                    if (rr, cc) not in self.obstacles and (rr, cc) not in self.dirts and (rr, cc) not in self.agent_positions:
                        self.agent_positions.append((rr, cc))
                        break
        else:
            data = LEVELS[idx]
            self.rows = data["rows"]
            self.cols = data["cols"]
            self.obstacles = set(data["obstacles"])
            self.dirts = set(data["dirts"])
            self.agent_positions.clear()
            if self.n_agents > 1:
                self.agent_positions.append(data["start"])
                for _ in range(self.n_agents - 1):
                    while True:
                        rr = random.randint(0, self.rows - 1)
                        cc = random.randint(0, self.cols - 1)
                        if (rr, cc) not in self.obstacles and (rr, cc) not in self.dirts and (rr, cc) not in self.agent_positions:
                            self.agent_positions.append((rr, cc))
                            break
            else:
                self.agent_positions.append(data["start"])
        for i in range(self.n_agents):
            self.done_flags[i] = False
        self.done = False
        self.score = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        for i in range(self.n_agents):
            self.energy[i] = self.energy_limit if self.energy_limit is not None else None

    def reset(self):
        if self.use_random:
            self.load_level(0)
        else:
            self.load_level(self.level_index)
        return self.get_state()

    def step(self, action):
        reward = -0.1
        if action == 4:
            if self.agent_positions[0] in self.dirts:
                self.dirts.remove(self.agent_positions[0])
                reward = 5.0
        else:
            r, c = self.agent_positions[0]
            nr, nc = r, c
            if action == 0:
                nr -= 1
            elif action == 1:
                nr += 1
            elif action == 2:
                nc -= 1
            elif action == 3:
                nc += 1
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or (nr, nc) in self.obstacles:
                reward = -5.0
            else:
                self.agent_positions[0] = (nr, nc)
        if len(self.dirts) == 0:
            reward += 10.0
            self.score += 1
            self.done = True
        self.episode_reward += reward
        self.episode_steps += 1
        if self.energy[0] is not None:
            self.energy[0] -= 1
            if self.energy[0] <= 0:
                self.done = True
        if self.time_limit is not None and self.episode_steps >= self.time_limit:
            self.done = True
        return self.get_state(), reward, self.done, {}

    def multi_step(self, agent_idx, action):
        if agent_idx < 0 or agent_idx >= self.n_agents:
            return 0.0, True
        if self.done_flags[agent_idx]:
            return 0.0, True
        r, c = self.agent_positions[agent_idx]
        rew = -0.1
        if action == 4:
            if (r, c) in self.dirts:
                self.dirts.remove((r, c))
                rew = 5.0
        else:
            nr, nc = r, c
            if action == 0:
                nr -= 1
            elif action == 1:
                nr += 1
            elif action == 2:
                nc -= 1
            elif action == 3:
                nc += 1
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or (nr, nc) in self.obstacles:
                rew = -5.0
            else:
                self.agent_positions[agent_idx] = (nr, nc)
        if len(self.dirts) == 0:
            rew += 10.0
            self.score += 1
            self.done_flags[agent_idx] = True
        self.episode_reward += rew
        self.episode_steps += 1
        if self.energy[agent_idx] is not None:
            self.energy[agent_idx] -= 1
            if self.energy[agent_idx] <= 0:
                self.done_flags[agent_idx] = True
        if self.time_limit is not None and self.episode_steps >= self.time_limit:
            self.done_flags[agent_idx] = True
        all_done = all(self.done_flags)
        return rew, all_done

    def step_all(self, actions):
        rewards = [0.0]*self.n_agents
        for i in range(self.n_agents):
            if not self.done_flags[i]:
                r, doneflag = self.multi_step(i, actions[i])
                rewards[i] = r
        return self.get_multi_state(), rewards, all(self.done_flags), {}

    def get_state(self):
        arr = [self.level_index, self.agent_positions[0][0], self.agent_positions[0][1]]
        dirt_map = np.zeros((self.rows, self.cols), dtype=int)
        for (dr, dc) in self.dirts:
            dirt_map[dr, dc] = 1
        return np.concatenate([np.array(arr, dtype=int), dirt_map.flatten()])

    def get_multi_state(self):
        st = [self.level_index]
        for (r, c) in self.agent_positions:
            st.append(r)
            st.append(c)
        dirt_map = np.zeros((self.rows, self.cols), dtype=int)
        for (dr, dc) in self.dirts:
            dirt_map[dr, dc] = 1
        return np.concatenate([np.array(st, dtype=int), dirt_map.flatten()])

    def next_level(self):
        if self.use_random:
            return
        self.level_index += 1
        if self.level_index >= self.max_level:
            self.level_index = 0
        self.load_level(self.level_index)

    def get_info(self):
        info = {
            "level_index": self.level_index,
            "score": self.score,
            "episode_reward": self.episode_reward,
            "episode_steps": self.episode_steps,
            "rows": self.rows,
            "cols": self.cols,
            "done_single": self.done,
            "done_flags": list(self.done_flags),
            "agent_positions": list(self.agent_positions)
        }
        energ = []
        for i in range(self.n_agents):
            energ.append(self.energy[i] if self.energy[i] is not None else None)
        info["energy_list"] = energ
        return info

    def close(self):
        pass

    def debug_print_map(self):
        row_strings = []
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if any((ar == r and ac == c) for (ar, ac) in self.agent_positions):
                    row_str += "A"
                elif (r, c) in self.obstacles:
                    row_str += "#"
                elif (r, c) in self.dirts:
                    row_str += "*"
                else:
                    row_str += "."
            row_strings.append(row_str)
        return row_strings

    def mouse_edit(self, rr, cc):
        if (rr, cc) in self.obstacles:
            self.obstacles.remove((rr, cc))
        else:
            if (rr, cc) not in self.dirts and all((pos != (rr, cc)) for pos in self.agent_positions):
                self.obstacles.add((rr, cc))

    def partial_observe(self, agent_idx, radius=1):
        if agent_idx < 0 or agent_idx >= self.n_agents:
            return None
        (r, c) = self.agent_positions[agent_idx]
        top = max(0, r - radius)
        bot = min(self.rows - 1, r + radius)
        left = max(0, c - radius)
        right = min(self.cols - 1, c + radius)
        sub_map = []
        for rr in range(top, bot + 1):
            row = []
            for cc in range(left, right + 1):
                if (rr, cc) == (r, c):
                    row.append("A")
                elif (rr, cc) in self.obstacles:
                    row.append("#")
                elif (rr, cc) in self.dirts:
                    row.append("*")
                else:
                    row.append(".")
            sub_map.append(row)
        return sub_map
