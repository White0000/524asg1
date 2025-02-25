import pygame
import sys
import math
import time
import random
import numpy as np
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.figure as mfigure

class SmoothAgentSprite:
    def __init__(self, idx, tile_size):
        self.idx = idx
        self.tile_size = tile_size
        self.draw_x = 0
        self.draw_y = 0
        self.target_x = 0
        self.target_y = 0
        self.speed = 999  # 若要瞬移，可设为999或直接改 update() 不插值

    def set_position(self, px, py):
        self.target_x = px
        self.target_y = py

    def instantly_move(self, px, py):
        self.draw_x = px
        self.draw_y = py
        self.target_x = px
        self.target_y = py

    def update(self):
        dx = self.target_x - self.draw_x
        dy = self.target_y - self.draw_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-3:
            # 已经足够近，直接贴齐
            self.draw_x = self.target_x
            self.draw_y = self.target_y
        else:
            # 做一次小步动画
            step = self.speed
            angle = math.atan2(dy, dx)
            mx = step * math.cos(angle)
            my = step * math.sin(angle)
            if abs(mx) > abs(dx):
                mx = dx
            if abs(my) > abs(dy):
                my = dy
            self.draw_x += mx
            self.draw_y += my


class PygameUI:
    def __init__(self, width=1280, height=1000, title="RoboCleaner - PyGame Enhanced", tile_size=48):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.title = title

        self.pause = False
        self.state = "MENU"

        self.selected_algo = None
        self.algorithms = ["Q-Learning","SARSA","DQN","PPO"]
        self.metrics = ["Reward","Loss","Epsilon"]
        self.selected_metric = "Reward"
        self.selected_metric2 = "Loss"
        self.multi_curve = False
        self.metric_data = []
        self.metric_data2 = []

        self.menu_buttons = []
        self.train_buttons = []

        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.screen = None
        self.clock = None

        self.side_panel_width = 420
        self.main_area_rect = None
        self.side_area_rect = None
        self.param_area_height = 350
        self.button_area_height = 330
        self.matplot_area_height = 350
        self.side_margin = 5

        self.env = None
        self.agent_count = 1
        self.smooth_sprites = []
        self.hover_info = None
        self.map_images = {}

        self.fig = None
        self.canvas_agg = None
        self.ax = None
        self.curve_surf = None

        self.last_loss = 0.0
        self.last_reward = 0.0
        self.last_epsilon = 0.0

        self.fast_forward = 1
        self.train_loops = 1
        self.show_grid = True
        self.show_names = False

        self.build_setup()

    def build_setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Arial", 40)
        self.font_medium = pygame.font.SysFont("Arial", 26)
        self.font_small = pygame.font.SysFont("Arial", 18)

        self.main_area_rect = pygame.Rect(0, 0, self.width - self.side_panel_width, self.height)
        self.side_area_rect = pygame.Rect(self.width - self.side_panel_width, 0, self.side_panel_width, self.height)

        self.create_menu_buttons()
        self.create_train_buttons()
        self.load_textures()
        self.init_matplotlib()

    def create_menu_buttons(self):
        cx = (self.width - self.side_panel_width)//2
        cy = self.height//2 - 120
        w = 360
        h = 70
        gap = 25
        for i, algo in enumerate(self.algorithms):
            rect = pygame.Rect(cx - w//2, cy + i*(h+gap), w, h)
            self.menu_buttons.append((algo, rect))

    def create_train_buttons(self):
        px = self.side_area_rect.x + self.side_margin
        py = self.side_area_rect.y + self.param_area_height + self.side_margin
        bw = 100
        bh = 40
        gap = 10

        pause_rect  = pygame.Rect(px,              py,          bw, bh)
        resume_rect = pygame.Rect(px+bw+gap,       py,          bw, bh)
        metric_rect = pygame.Rect(px,              py+bh+gap,   bw*2+gap, bh)
        menu_rect   = pygame.Rect(px,              metric_rect.y+bh+gap, bw, bh)
        speedp_rect = pygame.Rect(menu_rect.x+bw+gap, menu_rect.y, bw, bh)
        speedm_rect = pygame.Rect(speedp_rect.x+bw+gap, menu_rect.y, bw, bh)
        trainp_rect = pygame.Rect(px,              menu_rect.y+bh+gap, bw, bh)
        trainm_rect = pygame.Rect(trainp_rect.x+bw+gap, trainp_rect.y, bw, bh)
        grid_rect   = pygame.Rect(px,              trainp_rect.y+bh+gap, bw, bh)
        names_rect  = pygame.Rect(grid_rect.x+bw+gap, grid_rect.y, bw, bh)

        self.train_buttons.append(("Pause",    pause_rect))
        self.train_buttons.append(("Continue", resume_rect))
        self.train_buttons.append(("Metric",   metric_rect))
        self.train_buttons.append(("Menu",     menu_rect))
        self.train_buttons.append(("Speed++",  speedp_rect))
        self.train_buttons.append(("Speed--",  speedm_rect))
        self.train_buttons.append(("Train++",  trainp_rect))
        self.train_buttons.append(("Train--",  trainm_rect))
        self.train_buttons.append(("Grid",     grid_rect))
        self.train_buttons.append(("Names",    names_rect))

    def load_textures(self):
        try:
            floor_img = pygame.image.load("aset/floor.png").convert()
            floor_img = pygame.transform.scale(floor_img, (self.tile_size, self.tile_size))
            self.map_images["floor"] = floor_img
        except:
            self.map_images["floor"] = None

        try:
            stone_img = pygame.image.load("aset/stone.png").convert()
            stone_img = pygame.transform.scale(stone_img, (self.tile_size, self.tile_size))
            self.map_images["obstacle"] = stone_img
        except:
            self.map_images["obstacle"] = None

        try:
            dirt_img = pygame.image.load("aset/dirt.png").convert()
            dirt_img = pygame.transform.scale(dirt_img, (self.tile_size, self.tile_size))
            self.map_images["dirt"] = dirt_img
        except:
            self.map_images["dirt"] = None

        # agent(单个)
        try:
            ai_img = pygame.image.load("aset/ai.gif").convert()
            ai_img = pygame.transform.scale(ai_img, (self.tile_size, self.tile_size))
            self.map_images["agent"] = ai_img
        except:
            fallback = pygame.Surface((self.tile_size, self.tile_size))
            fallback.fill((0, 0, 255))
            self.map_images["agent"] = fallback

    def init_matplotlib(self):
        self.fig = mfigure.Figure(figsize=(3.65, 2.55), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.18, bottom=0.25, right=0.95, top=0.88)
        self.canvas_agg = agg.FigureCanvasAgg(self.fig)
        self.curve_surf = None

    def update_matplot_curve(self):
        self.ax.clear()
        vals = self.metric_data[-500:]
        xdat = list(range(len(vals)))
        self.ax.plot(xdat, vals, color="lightskyblue", label=self.selected_metric)
        if self.multi_curve and len(self.metric_data2)>1:
            vals2 = self.metric_data2[-500:]
            self.ax.plot(xdat, vals2, color="red", label=self.selected_metric2)
        self.ax.set_xlim(0, 500)
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Training Metrics")
        self.ax.legend(loc="upper left")
        self.ax.grid(True)
        self.canvas_agg.draw()
        raw_data = self.canvas_agg.buffer_rgba()
        sz = self.canvas_agg.get_width_height()
        self.curve_surf = pygame.image.frombuffer(raw_data, sz, "RGBA")

    def render_matplot_curve(self):
        if self.curve_surf:
            gx = self.side_area_rect.x + self.side_margin
            gy = self.side_area_rect.y + self.param_area_height + self.button_area_height + self.side_margin
            self.screen.blit(self.curve_surf, (gx, gy))

    def attach_environment(self, env):
        self.env = env
        if hasattr(env, "agent_positions"):
            self.agent_count = len(env.agent_positions)
        else:
            self.agent_count = 1
        self.init_smooth_sprites()

    def init_smooth_sprites(self):
        self.smooth_sprites = []
        if not self.env or not hasattr(self.env,"agent_positions"):
            return
        for i in range(self.agent_count):
            sp = SmoothAgentSprite(i, self.tile_size)
            r, c = self.env.agent_positions[i]
            px, py = self.grid_to_px(r, c)
            sp.instantly_move(px, py)
            self.smooth_sprites.append(sp)

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.pause = not self.pause
                elif event.key == pygame.K_m:
                    self.switch_metric()
                elif event.key == pygame.K_c:
                    self.multi_curve = not self.multi_curve

            if self.state == "MENU":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    for algo, rect in self.menu_buttons:
                        if rect.collidepoint(mx, my):
                            self.selected_algo = algo
                            self.state="TRAIN"

            elif self.state == "TRAIN":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if event.button == 3:
                        gx = (mx-20)//self.tile_size
                        gy = (my-20)//self.tile_size
                        if self.env and hasattr(self.env,"mouse_edit"):
                            self.env.mouse_edit(gy,gx)
                    else:
                        for name, rect in self.train_buttons:
                            if rect.collidepoint(mx, my):
                                if name=="Pause":
                                    self.pause=True
                                elif name=="Continue":
                                    self.pause=False
                                elif name=="Metric":
                                    self.switch_metric()
                                elif name=="Menu":
                                    self.state="MENU"
                                elif name=="Speed++":
                                    self.fast_forward=min(100,self.fast_forward*2)
                                elif name=="Speed--":
                                    self.fast_forward=max(1,self.fast_forward//2)
                                elif name=="Train++":
                                    self.train_loops=min(100,self.train_loops+1)
                                elif name=="Train--":
                                    self.train_loops=max(1,self.train_loops-1)
                                elif name=="Grid":
                                    self.show_grid=not self.show_grid
                                elif name=="Names":
                                    self.show_names=not self.show_names

        # Hover
        mx,my = pygame.mouse.get_pos()
        gx = (mx-20)//self.tile_size
        gy = (my-20)//self.tile_size
        self.hover_info = None
        if self.state=="TRAIN" and self.env:
            if 0<=gy<self.env.rows and 0<=gx<self.env.cols:
                info = f"Cell=({gy},{gx}) "
                if (gy,gx) in self.env.obstacles:
                    info += "Obstacle "
                elif (gy,gx) in self.env.dirts:
                    info += "Dirt "
                self.hover_info = info

    def switch_metric(self):
        if self.selected_metric=="Reward":
            self.selected_metric="Loss"
        elif self.selected_metric=="Loss":
            self.selected_metric="Epsilon"
        else:
            self.selected_metric="Reward"

    def render(self, environment, loss=0.0, reward=0.0, epsilon=0.0):
        self.last_loss   = loss
        self.last_reward = reward
        self.last_epsilon= epsilon
        self.screen.fill((34,36,50))

        if self.state == "MENU":
            self.render_menu()
        else:
            self.render_training(environment)

        pygame.display.flip()

    def render_menu(self):
        pygame.draw.rect(self.screen,(44,46,60), self.main_area_rect)
        pygame.draw.rect(self.screen,(70,70,90), self.side_area_rect)

        txt = self.font_large.render("Select Algorithm - PyGame", True, (255,255,255))
        self.screen.blit(txt, (self.main_area_rect.centerx-240,
                               self.main_area_rect.centery-200))

        mx,my = pygame.mouse.get_pos()
        for algo, rect in self.menu_buttons:
            col = (100,100,130)
            if rect.collidepoint(mx,my):
                col = (140,140,170)
            pygame.draw.rect(self.screen, col, rect, border_radius=8)
            t = self.font_medium.render(algo, True, (255,255,255))
            self.screen.blit(t, (rect.x+10, rect.y+10))

    def render_training(self, environment):
        pygame.draw.rect(self.screen,(20,22,30), self.main_area_rect)
        pygame.draw.rect(self.screen,(58,60,75), self.side_area_rect)

        line_y1 = self.side_area_rect.y + self.param_area_height
        line_y2 = line_y1 + self.button_area_height
        pygame.draw.line(self.screen,(150,150,150),
                         (self.side_area_rect.x, line_y1),
                         (self.side_area_rect.x + self.side_panel_width, line_y1), 2)
        pygame.draw.line(self.screen,(150,150,150),
                         (self.side_area_rect.x, line_y2),
                         (self.side_area_rect.x + self.side_panel_width, line_y2), 2)

        self.update_smooth_positions()
        self.render_map(environment)
        self.render_side_panel(environment)
        self.update_metrics()
        self.update_matplot_curve()
        self.render_matplot_curve()
        self.show_hover()

    def render_map(self, environment):
        total_map_w = environment.cols * self.tile_size
        total_map_h = environment.rows * self.tile_size
        map_x = (self.main_area_rect.width - total_map_w)//2
        map_y = (self.main_area_rect.height - total_map_h)//2
        ox = self.main_area_rect.x + map_x
        oy = self.main_area_rect.y + map_y

        # 绘制地图
        for r in range(environment.rows):
            for c in range(environment.cols):
                x = ox + c*self.tile_size
                y = oy + r*self.tile_size

                if (r,c) in environment.obstacles:
                    # stone
                    img = self.map_images["obstacle"]
                    if img:
                        self.screen.blit(img,(x,y))
                    else:
                        pygame.draw.rect(self.screen,(110,110,110),(x,y,self.tile_size,self.tile_size))
                elif (r,c) in environment.dirts:
                    # dirt
                    img = self.map_images["dirt"]
                    if img:
                        self.screen.blit(img,(x,y))
                    else:
                        pygame.draw.rect(self.screen,(139,69,19),(x,y,self.tile_size,self.tile_size))
                else:
                    # floor
                    img = self.map_images["floor"]
                    if img:
                        self.screen.blit(img,(x,y))
                    else:
                        pygame.draw.rect(self.screen,(200,200,200),(x,y,self.tile_size,self.tile_size))

                if self.show_grid:
                    pygame.draw.rect(self.screen, (20,20,30), (x,y,self.tile_size,self.tile_size), 1)
                if self.show_names:
                    nm_t = self.font_small.render(f"{r},{c}", True, (80,80,255))
                    self.screen.blit(nm_t, (x+2,y+2))

        # 绘制单个机器人(或多)
        for sp in self.smooth_sprites:
            ai_surf = self.map_images["agent"]  # 这里是单一Surface
            drawx = sp.draw_x  # 移除 -20 偏移
            drawy = sp.draw_y
            # 再加上地图起点 ox, oy
            rx = ox + drawx
            ry = oy + drawy
            if ai_surf:
                self.screen.blit(ai_surf, (rx, ry))
            else:
                pygame.draw.rect(self.screen, (30,144,255),
                                 (rx, ry, self.tile_size, self.tile_size))

    def render_side_panel(self, environment):
        px = self.side_area_rect.x + self.side_margin
        py = self.side_area_rect.y + self.side_margin

        algo_txt = self.font_medium.render("Algorithm: "+str(self.selected_algo), True, (255,255,0))
        self.screen.blit(algo_txt,(px,py))
        py += 35
        if hasattr(environment, "level_index") and hasattr(environment, "max_level"):
            ltxt = f"Level: {environment.level_index+1}/{environment.max_level}"
        else:
            ltxt="Level: ?/?"
        level_t = self.font_medium.render(ltxt,True,(255,100,100))
        self.screen.blit(level_t,(px,py))
        py += 30
        dcnt = len(environment.dirts) if hasattr(environment,"dirts") else 0
        dt = self.font_medium.render(f"Dirt: {dcnt}",True,(255,0,0))
        self.screen.blit(dt,(px,py))
        py += 30
        st = self.font_medium.render("Score: "+str(environment.score),True,(200,200,200))
        self.screen.blit(st,(px,py))
        py += 30
        ltx = self.font_medium.render("Loss: "+str(round(self.last_loss,3)),True,(0,255,0))
        self.screen.blit(ltx,(px,py))
        py += 30
        rtx = self.font_medium.render("Reward: "+str(round(self.last_reward,3)),True,(0,255,0))
        self.screen.blit(rtx,(px,py))
        py += 30
        etx = self.font_medium.render("Epsilon: "+str(round(self.last_epsilon,3)),True,(0,255,0))
        self.screen.blit(etx,(px,py))
        py += 30
        ff_text = self.font_medium.render("Speed x"+str(self.fast_forward), True,(255,255,0))
        self.screen.blit(ff_text,(px,py))
        py += 30
        tr_text = self.font_medium.render("Train Loops: "+str(self.train_loops),True,(255,255,0))
        self.screen.blit(tr_text,(px,py))
        py += 35

        if hasattr(environment, "energy"):
            en_t = self.font_medium.render("Energy: "+str(environment.energy), True, (255,255,255))
            self.screen.blit(en_t,(px,py))
            py += 25
        if hasattr(environment, "time_limit"):
            tl_t = self.font_medium.render("TimeLimit: "+str(environment.time_limit), True,(255,255,255))
            self.screen.blit(tl_t,(px,py))
            py += 25

        mx,my = pygame.mouse.get_pos()
        for name, rect in self.train_buttons:
            color=(200,0,0) if name=="Pause" else (0,200,0) if name=="Continue" else (0,0,200)
            if name=="Menu":
                color=(180,120,30)
            elif name=="Speed++":
                color=(150,150,0)
            elif name=="Speed--":
                color=(150,100,0)
            elif name=="Train++":
                color=(100,150,0)
            elif name=="Train--":
                color=(100,110,0)
            elif name=="Grid":
                color=(100,100,255)
            elif name=="Names":
                color=(120,80,220)

            if rect.collidepoint(mx,my):
                color=tuple(min(255,c+50) for c in color)

            pygame.draw.rect(self.screen,color,rect,border_radius=8)

            if name=="Pause":
                t=self.font_small.render("Pause",True,(255,255,255))
            elif name=="Continue":
                t=self.font_small.render("Continue",True,(255,255,255))
            elif name=="Metric":
                t=self.font_small.render("Metric: "+self.selected_metric,True,(255,255,255))
            elif name=="Menu":
                t=self.font_small.render("Menu",True,(255,255,255))
            elif name=="Speed++":
                t=self.font_small.render("Speed++",True,(255,255,255))
            elif name=="Speed--":
                t=self.font_small.render("Speed--",True,(255,255,255))
            elif name=="Train++":
                t=self.font_small.render("Train++",True,(255,255,255))
            elif name=="Train--":
                t=self.font_small.render("Train--",True,(255,255,255))
            else:
                t=self.font_small.render(name,True,(255,255,255))
            self.screen.blit(t,(rect.x+8, rect.y+8))

    def update_metrics(self):
        val = 0.0
        if self.selected_metric=="Loss":
            val = self.last_loss
        elif self.selected_metric=="Epsilon":
            val = self.last_epsilon
        else:
            val = self.last_reward
        self.metric_data.append(val)

        if self.multi_curve:
            alt = 0.0
            if self.selected_metric=="Reward":
                alt = self.last_loss
            elif self.selected_metric=="Loss":
                alt = self.last_reward
            else:
                alt = self.last_loss
            self.metric_data2.append(alt)

    def update_matplot_curve(self):
        if not self.metric_data:
            return
        self.ax.clear()
        xvals = list(range(len(self.metric_data[-500:])))
        self.ax.plot(xvals, self.metric_data[-500:], color="lightskyblue", label=self.selected_metric)

        if self.multi_curve and len(self.metric_data2) > 1:
            self.ax.plot(xvals, self.metric_data2[-500:], color="red", label=self.selected_metric2)

        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Training Metrics")
        self.ax.grid(True)
        self.ax.legend(loc="upper left")
        self.canvas_agg.draw()

        raw_data = self.canvas_agg.buffer_rgba()
        sz = self.canvas_agg.get_width_height()
        self.curve_surf = pygame.image.frombuffer(raw_data, sz, "RGBA")

    def render_matplot_curve(self):
        if self.curve_surf:
            gx = self.side_area_rect.x + self.side_margin
            gy = self.side_area_rect.y + self.param_area_height + self.button_area_height + self.side_margin
            self.screen.blit(self.curve_surf,(gx,gy))

    def show_hover(self):
        if self.hover_info:
            px = self.side_area_rect.x + self.side_margin
            py = self.side_area_rect.y + self.param_area_height + self.button_area_height - 30
            txt = self.font_small.render(self.hover_info,True,(255,255,0))
            self.screen.blit(txt,(px,py))

    def update_smooth_positions(self):
        if not self.env or not hasattr(self.env, "agent_positions"):
            return
        for sp in self.smooth_sprites:
            r, c = self.env.agent_positions[sp.idx]
            px, py = self.grid_to_px(r, c)
            sp.set_position(px, py)
            sp.update()

    def grid_to_px(self, r, c):
        # 不再做 -20 偏移；每个格子宽=tile_size
        return (c*self.tile_size, r*self.tile_size)

    def set_fps(self,fps):
        self.clock.tick(fps * self.fast_forward)

    def close(self):
        pygame.quit()

    def is_paused(self):
        return self.pause
