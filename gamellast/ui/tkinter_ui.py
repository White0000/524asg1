import tkinter as tk
import random
import math
import time
import sys
import numpy as np
from PIL import Image, ImageTk

class SmoothSprite:
    def __init__(self, idx, tile_size):
        self.idx = idx
        self.tile_size = tile_size
        self.draw_x = 0.0
        self.draw_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.speed = 0.25
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
        if dist<1e-3:
            self.draw_x = self.target_x
            self.draw_y = self.target_y
        else:
            step = self.speed
            ang = math.atan2(dy, dx)
            mx = step*math.cos(ang)
            my = step*math.sin(ang)
            if abs(mx)>abs(dx):
                mx=dx
            if abs(my)>abs(dy):
                my=dy
            self.draw_x+=mx
            self.draw_y+=my

class TkinterUI:
    def __init__(self, width=1200, height=900, title="RoboCleaner", tile_size=40):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(str(width) + "x" + str(height))
        self.state = "MENU"
        self.pause = False
        self.algorithms = ["Q-Learning", "SARSA", "DQN", "PPO"]
        self.selected_algo = None
        self.metrics = ["Reward","Loss","Epsilon"]
        self.selected_metric = "Reward"
        self.selected_metric2 = "Loss"
        self.multi_curve = False
        self.metric_data = []
        self.metric_data2 = []
        self.loss_val = 0.0
        self.reward_val = 0.0
        self.epsilon_val = 0.0
        self.environment = None
        self.smooth_sprites = []
        self.energy_display = None
        self.time_limit_display = None
        self.menu_frame = None
        self.settings_frame = None
        self.train_frame = None
        self.menu_buttons = []
        self.rand_var = tk.BooleanVar(value=False)
        self.energy_val = tk.StringVar()
        self.time_val = tk.StringVar()
        self.load_textures()
        self.setup_ui()

    def load_textures(self):
        self.img_floor=None
        self.img_obs=None
        self.img_dirt=None
        self.img_agents=[]
        try:
            floor_img=Image.open("../aset/floor.png")
            floor_img=floor_img.resize((self.tile_size,self.tile_size),Image.ANTIALIAS)
            self.img_floor=ImageTk.PhotoImage(floor_img)
        except:
            self.img_floor=None
        try:
            obs_img=Image.open("../aset/stone.png")
            obs_img=obs_img.resize((self.tile_size,self.tile_size),Image.ANTIALIAS)
            self.img_obs=ImageTk.PhotoImage(obs_img)
        except:
            self.img_obs=None
        try:
            dirt_img=Image.open("../aset/dirt.png")
            dirt_img=dirt_img.resize((self.tile_size,self.tile_size),Image.ANTIALIAS)
            self.img_dirt=ImageTk.PhotoImage(dirt_img)
        except:
            self.img_dirt=None
        for i in range(4):
            fn=f"robot{i}.png"
            try:
                aimg=Image.open(fn)
                aimg=aimg.resize((self.tile_size,self.tile_size),Image.ANTIALIAS)
                tkimg=ImageTk.PhotoImage(aimg)
                self.img_agents.append(tkimg)
            except:
                pass
        if not self.img_agents:
            self.img_agents=[None]

    def setup_ui(self):
        self.create_menu_frame()
        self.create_settings_frame()
        self.create_train_frame()
        self.show_menu()

    def attach_environment(self, env):
        self.environment = env
        self.init_sprites()

    def init_sprites(self):
        self.smooth_sprites.clear()
        if not self.environment:
            return
        if hasattr(self.environment,"agent_positions"):
            n=len(self.environment.agent_positions)
            for i in range(n):
                s=SmoothSprite(i,self.tile_size)
                r,c=self.environment.agent_positions[i]
                px,py=self.grid_to_px(r,c)
                s.instantly_move(px,py)
                self.smooth_sprites.append(s)

    def grid_to_px(self, r, c):
        ox=50
        oy=50
        return (ox+c*self.tile_size, oy+r*self.tile_size)

    def create_menu_frame(self):
        self.menu_frame = tk.Frame(self.root, bg="#222222")
        label = tk.Label(self.menu_frame, text="Select Algorithm", fg="white", bg="#222222", font=("Arial",24))
        label.pack(pady=50)
        for algo in self.algorithms:
            btn = tk.Button(self.menu_frame, text=algo, fg="white", bg="#444444", font=("Arial",16),
                            command=lambda a=algo: self.select_algo(a))
            btn.pack(pady=10, ipadx=40, ipady=10)

    def create_settings_frame(self):
        self.settings_frame = tk.Frame(self.root, bg="#333333")
        stitle = tk.Label(self.settings_frame, text="Settings", fg="white", bg="#333333", font=("Arial",20))
        stitle.pack(pady=20)
        cbox = tk.Checkbutton(self.settings_frame, text="Use Random Map", variable=self.rand_var,
                              fg="white", bg="#333333", selectcolor="#444444", font=("Arial",14))
        cbox.pack(pady=10)
        eframe = tk.Frame(self.settings_frame, bg="#333333")
        eframe.pack(pady=5)
        elabel = tk.Label(eframe, text="Energy Limit:", fg="white", bg="#333333", font=("Arial",14))
        elabel.pack(side=tk.LEFT, padx=5)
        self.energy_val.set("")
        eentry = tk.Entry(eframe, textvariable=self.energy_val, width=10, font=("Arial",14))
        eentry.pack(side=tk.LEFT, padx=5)
        tframe = tk.Frame(self.settings_frame, bg="#333333")
        tframe.pack(pady=5)
        tlabel = tk.Label(tframe, text="Time Limit:", fg="white", bg="#333333", font=("Arial",14))
        tlabel.pack(side=tk.LEFT, padx=5)
        self.time_val.set("")
        tentry = tk.Entry(tframe, textvariable=self.time_val, width=10, font=("Arial",14))
        tentry.pack(side=tk.LEFT, padx=5)
        cbtn = tk.Button(self.settings_frame, text="Confirm", fg="white", bg="green", font=("Arial",16),
                         command=self.confirm_settings)
        cbtn.pack(pady=30)

    def create_train_frame(self):
        self.train_frame = tk.Frame(self.root, bg="black")
        topbar = tk.Frame(self.train_frame, bg="#333333", height=60)
        topbar.pack(side=tk.TOP, fill=tk.X)
        self.pause_button = tk.Button(topbar, text="Pause", fg="white", bg="red", font=("Arial",12),
                                      command=self.on_pause_click)
        self.pause_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.metric_button = tk.Button(topbar, text="Metric: Reward", fg="white", bg="blue", font=("Arial",12),
                                       command=self.on_metric_click)
        self.metric_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas_width = self.width
        self.canvas_height = self.height-60
        self.train_canvas = tk.Canvas(self.train_frame, width=self.canvas_width, height=self.canvas_height, bg="#000000")
        self.train_canvas.pack(fill=tk.BOTH, expand=True)
        self.train_canvas.bind("<Button-3>", self.right_click)
        self.train_canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Key>", self.on_key_press)

    def select_algo(self, algo):
        self.selected_algo = algo
        self.hide_menu()
        self.show_settings()
        self.state = "SETTINGS"

    def confirm_settings(self):
        self.hide_settings()
        self.state = "TRAIN"

    def on_pause_click(self):
        self.pause = not self.pause
        if self.pause:
            self.pause_button.config(text="Continue", bg="green")
        else:
            self.pause_button.config(text="Pause", bg="red")

    def on_metric_click(self):
        idx = self.metrics.index(self.selected_metric)
        idx = (idx+1) % len(self.metrics)
        self.selected_metric = self.metrics[idx]
        self.metric_button.config(text="Metric: " + self.selected_metric)

    def show_menu(self):
        self.menu_frame.pack(fill=tk.BOTH, expand=True)

    def hide_menu(self):
        self.menu_frame.pack_forget()

    def show_settings(self):
        self.settings_frame.pack(fill=tk.BOTH, expand=True)

    def hide_settings(self):
        self.settings_frame.pack_forget()

    def show_train(self):
        self.train_frame.pack(fill=tk.BOTH, expand=True)

    def hide_train(self):
        self.train_frame.pack_forget()

    def process_events(self):
        self.root.update_idletasks()
        self.root.update()
        if self.state == "MENU":
            pass
        elif self.state == "SETTINGS":
            pass
        elif self.state == "TRAIN":
            self.hide_settings()
            self.show_train()
            if not self.pause:
                self.update_sprites()
            self.update_train_canvas()

    def attach_environment_data(self, env):
        self.environment=env
        self.init_sprites()

    def init_sprites(self):
        self.smooth_sprites.clear()
        if not self.environment:
            return
        if hasattr(self.environment,"agent_positions"):
            for i,(rr,cc) in enumerate(self.environment.agent_positions):
                sp=SmoothSprite(i,self.tile_size)
                px,py=self.grid_to_px(rr,cc)
                sp.instantly_move(px,py)
                self.smooth_sprites.append(sp)

    def update_sprites(self):
        if not self.environment or not hasattr(self.environment,"agent_positions"):
            return
        for i in range(len(self.smooth_sprites)):
            rr,cc=self.environment.agent_positions[i]
            px,py=self.grid_to_px(rr,cc)
            self.smooth_sprites[i].set_position(px,py)
            self.smooth_sprites[i].update()

    def right_click(self, event):
        if not self.environment:
            return
        gx=(event.x-50)//self.tile_size
        gy=(event.y-50)//self.tile_size
        if hasattr(self.environment,"mouse_edit"):
            if 0<=gy<self.environment.rows and 0<=gx<self.environment.cols:
                self.environment.mouse_edit(gy,gx)
        self.init_sprites()

    def on_mouse_move(self, event):
        pass

    def on_key_press(self, event):
        if event.char.lower()=="p":
            self.on_pause_click()
        elif event.char.lower()=="m":
            self.on_metric_click()
        elif event.char.lower()=="c":
            self.multi_curve=not self.multi_curve

    def grid_to_px(self, r,c):
        ox=50
        oy=50
        return (ox+c*self.tile_size, oy+r*self.tile_size)

    def render(self, environment, loss=0.0, reward=0.0, epsilon=0.0):
        self.environment=environment
        self.loss_val=loss
        self.reward_val=reward
        self.epsilon_val=epsilon
        val=0.0
        if self.selected_metric=="Loss":
            val=self.loss_val
        elif self.selected_metric=="Reward":
            val=self.reward_val
        else:
            val=self.epsilon_val
        self.metric_data.append(val)
        if len(self.metric_data)>200:
            self.metric_data.pop(0)
        if self.multi_curve:
            alt=0.0
            if self.selected_metric=="Loss":
                alt=self.reward_val
            else:
                alt=self.loss_val
            self.metric_data2.append(alt)
            if len(self.metric_data2)>200:
                self.metric_data2.pop(0)

    def update_train_canvas(self):
        self.train_canvas.delete("all")
        if not self.environment:
            return
        self.draw_world()
        self.draw_agents()
        self.draw_info()
        self.draw_metric_graph()

    def draw_world(self):
        offset_x=50
        offset_y=50
        for r in range(self.environment.rows):
            for c in range(self.environment.cols):
                x=offset_x+c*self.tile_size
                y=offset_y+r*self.tile_size
                fill_color="white"
                if (r,c) in self.environment.obstacles:
                    fill_color="#646464"
                elif (r,c) in self.environment.dirts:
                    fill_color="#8B4513"
                self.train_canvas.create_rectangle(x,y,x+self.tile_size,y+self.tile_size,
                                                   fill=fill_color, outline="black")

    def draw_agents(self):
        for i,sp in enumerate(self.smooth_sprites):
            self.train_canvas.create_rectangle(sp.draw_x,sp.draw_y,
                                               sp.draw_x+self.tile_size,sp.draw_y+self.tile_size,
                                               fill="#1E90FF", outline="black")

    def draw_info(self):
        algo_text = f"Algorithm: {self.selected_algo}"
        lvl_txt="?"
        if hasattr(self.environment,"level_index") and hasattr(self.environment,"max_level"):
            lvl_txt=f"Level: {self.environment.level_index+1}/{self.environment.max_level}"
        dirt_left=len(self.environment.dirts) if hasattr(self.environment,"dirts") else 0
        dirt_text=f"Dirt: {dirt_left}"
        score_text="?"
        if hasattr(self.environment,"score"):
            score_text=f"Score: {self.environment.score}"
        lines=[algo_text,lvl_txt,dirt_text,score_text,
               f"Loss: {round(self.loss_val,3)}",
               f"Reward: {round(self.reward_val,3)}",
               f"Epsilon: {round(self.epsilon_val,3)}"]
        if hasattr(self.environment,"energy") and self.environment.energy is not None:
            lines.append(f"Energy: {self.environment.energy}")
        if hasattr(self.environment,"time_limit") and self.environment.time_limit is not None:
            lines.append(f"TimeLimit: {self.environment.time_limit}")
        ybase=20
        for txt in lines:
            self.train_canvas.create_text(10, ybase, text=txt, fill="yellow", anchor="w", font=("Arial",14))
            ybase+=25
        if self.pause:
            self.train_canvas.create_text(10, ybase, text="[PAUSED]", fill="red", anchor="w", font=("Arial",14))

    def draw_metric_graph(self):
        if len(self.metric_data)<2:
            return
        graph_left=self.width//2
        graph_top=self.height-200
        graph_width=self.width//2-60
        graph_height=150
        self.train_canvas.create_rectangle(graph_left,graph_top,graph_left+graph_width,graph_top+graph_height, fill="#333333")
        vals=self.metric_data[-100:]
        mx=max(vals) if vals else 1
        if mx<=0:
            mx=1
        step=graph_width/(len(vals)-1)
        scale=graph_height/mx
        px=graph_left
        py=graph_top+graph_height-vals[0]*scale
        for i in range(1,len(vals)):
            nx=graph_left+i*step
            ny=graph_top+graph_height-vals[i]*scale
            self.train_canvas.create_line(px,py,nx,ny,fill="green",width=2)
            px,py=nx,ny
        if self.multi_curve and len(self.metric_data2)>1:
            vals2=self.metric_data2[-100:]
            mx2=max(vals2) if vals2 else 1
            if mx2<=0:
                mx2=1
            step2=graph_width/(len(vals2)-1)
            scale2=graph_height/mx2
            px2=graph_left
            py2=graph_top+graph_height-vals2[0]*scale2
            for i in range(1,len(vals2)):
                nx2=graph_left+i*step2
                ny2=graph_top+graph_height-vals2[i]*scale2
                self.train_canvas.create_line(px2,py2,nx2,ny2,fill="red",width=2)
                px2,py2=nx2,ny2

    def set_fps(self, fps):
        pass

    def close(self):
        self.root.destroy()

    def is_paused(self):
        return self.pause
