import json
import random
import os

LEVELS = []

def create_level(rows, cols, obstacles, dirts, start, name="CustomLevel", difficulty=1, energy=100, time_limit=100):
    return {
        "name": name,
        "difficulty": difficulty,
        "rows": rows,
        "cols": cols,
        "obstacles": obstacles,
        "dirts": dirts,
        "start": start,
        "energy": energy,
        "time_limit": time_limit
    }

def add_level(rows, cols, obstacles, dirts, start, name="NewLevel", difficulty=1, energy=100, time_limit=100):
    LEVELS.append(create_level(rows, cols, obstacles, dirts, start, name, difficulty, energy, time_limit))

def generate_random_level(rows, cols, obstacle_count, dirt_count, name="RandomLevel", difficulty=2):
    obstacles = set()
    while len(obstacles) < obstacle_count:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        obstacles.add((r, c))
    dirts = set()
    while len(dirts) < dirt_count:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in obstacles:
            dirts.add((r, c))
    if difficulty <= 2:
        energy = 120
        time_limit = 150
    elif difficulty == 3:
        energy = 80
        time_limit = 120
    else:
        energy = 50
        time_limit = 100
    start = (0, 0)
    lv = create_level(rows, cols, list(obstacles), list(dirts), start, name, difficulty, energy, time_limit)
    LEVELS.append(lv)

def load_custom_levels_from_file(path):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for lv in data:
            if all(k in lv for k in ["rows","cols","obstacles","dirts","start"]):
                LEVELS.append(lv)

def save_levels_to_file(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(LEVELS, f, indent=2)

def get_level_data(index):
    if index < 0 or index >= len(LEVELS):
        return None
    return LEVELS[index]

def set_level_data(index, new_data):
    if 0 <= index < len(LEVELS):
        LEVELS[index] = new_data

def remove_level(index):
    if 0 <= index < len(LEVELS):
        LEVELS.pop(index)

def swap_levels(idx_a, idx_b):
    if 0 <= idx_a < len(LEVELS) and 0 <= idx_b < len(LEVELS):
        LEVELS[idx_a], LEVELS[idx_b] = LEVELS[idx_b], LEVELS[idx_a]

def build_random_five_levels():
    difficulties = [1, 2, 3, 4, 5]
    for i, diff in enumerate(difficulties, start=1):
        rows = 5 + diff
        cols = 5 + diff
        obs_count = 4 + diff*2
        dirt_count = 3 + diff
        generate_random_level(rows, cols, obs_count, dirt_count, f"RandomLevel{i}", diff)

build_random_five_levels()
