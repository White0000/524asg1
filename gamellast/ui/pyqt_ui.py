import sys
import math
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QCheckBox, QLineEdit, QStackedWidget, QMessageBox, QFrame
)
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QPixmap
from PyQt5.QtCore import Qt, QRect

class SmoothSprite:
    def __init__(self, index, tile_size):
        self.index = index
        self.tile_size = tile_size
        self.draw_x = 0.0
        self.draw_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.speed = 0.3
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
            self.draw_x = self.target_x
            self.draw_y = self.target_y
        else:
            step = self.speed
            angle = math.atan2(dy, dx)
            mx = step*math.cos(angle)
            my = step*math.sin(angle)
            if abs(mx)>abs(dx):
                mx=dx
            if abs(my)>abs(dy):
                my=dy
            self.draw_x += mx
            self.draw_y += my

class PyQtUI:
    def __init__(self, width=1200, height=900, title="RoboCleaner", tile_size=40):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.title = title
        self.state = "MENU"
        self.pause = False
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle(self.title)
        self.window.setGeometry(100, 100, width, height)
        self.stack = QStackedWidget()
        self.menu_widget = None
        self.settings_widget = None
        self.train_widget = None
        self.selected_algo = None
        self.algorithms = ["Q-Learning", "SARSA", "DQN", "PPO"]
        self.use_random_map = False
        self.energy_limit = None
        self.time_limit = None
        self.metrics = ["Reward", "Loss", "Epsilon"]
        self.selected_metric = "Reward"
        self.metric_data = []
        self.loss_val = 0.0
        self.reward_val = 0.0
        self.eps_val = 0.0
        self.environment = None
        self.build_interface()
        self.window.setCentralWidget(self.stack)
        self.window.show()

    def build_interface(self):
        self.menu_widget = MenuWidget(self.algorithms, self.on_algo_selected)
        self.settings_widget = SettingsWidget(self.on_settings_changed, self.on_settings_confirm)
        self.train_widget = TrainWidget(
            self.tile_size, self.metrics, self.selected_metric,
            self.on_pause_toggle, self.on_metric_changed
        )
        self.stack.addWidget(self.menu_widget)
        self.stack.addWidget(self.settings_widget)
        self.stack.addWidget(self.train_widget)
        self.stack.setCurrentWidget(self.menu_widget)

    def attach_environment(self, env):
        self.environment = env
        if self.train_widget:
            self.train_widget.attach_environment(env)

    def on_algo_selected(self, algo):
        self.selected_algo = algo
        self.stack.setCurrentWidget(self.settings_widget)
        self.state = "SETTINGS"

    def on_settings_changed(self, use_rand, energy_val, time_val):
        self.use_random_map = use_rand
        if energy_val.lower()=="none" or energy_val=="":
            self.energy_limit = None
        else:
            try:
                self.energy_limit = int(energy_val)
            except ValueError:
                self.energy_limit = None
        if time_val.lower()=="none" or time_val=="":
            self.time_limit = None
        else:
            try:
                self.time_limit = int(time_val)
            except ValueError:
                self.time_limit = None

    def on_settings_confirm(self):
        self.stack.setCurrentWidget(self.train_widget)
        self.state = "TRAIN"

    def on_pause_toggle(self):
        self.pause = not self.pause
        if self.train_widget:
            self.train_widget.pause = self.pause

    def on_metric_changed(self, metric):
        self.selected_metric = metric
        if self.train_widget:
            self.train_widget.selected_metric = metric

    def process_events(self):
        self.app.processEvents()

    def render(self, environment, loss=0.0, reward=0.0, epsilon=0.0):
        self.environment = environment
        self.loss_val = loss
        self.reward_val = reward
        self.eps_val = epsilon
        if self.state=="TRAIN":
            val = 0.0
            if self.selected_metric=="Loss":
                val=self.loss_val
            elif self.selected_metric=="Reward":
                val=self.reward_val
            elif self.selected_metric=="Epsilon":
                val=self.eps_val
            self.metric_data.append(val)
            if len(self.metric_data)>200:
                self.metric_data.pop(0)
            if self.train_widget:
                self.train_widget.set_env_data(environment, self.loss_val, self.reward_val, self.eps_val)
                self.train_widget.update()

    def set_fps(self, fps):
        pass

    def close(self):
        self.window.close()

    def exec(self):
        sys.exit(self.app.exec_())

    def is_paused(self):
        return self.pause

class MenuWidget(QWidget):
    def __init__(self, algorithms, callback):
        super().__init__()
        self.algorithms = algorithms
        self.callback = callback
        self.layout = QVBoxLayout()
        self.title_label = QLabel("Select Algorithm - PyQt")
        f = self.title_label.font()
        f.setPointSize(24)
        self.title_label.setFont(f)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)
        for algo in self.algorithms:
            btn = QPushButton(algo)
            btn.setFixedSize(200, 50)
            btn.clicked.connect(lambda _, a=algo: self.select_algo(a))
            self.layout.addWidget(btn, 0, Qt.AlignCenter)
        self.setLayout(self.layout)

    def select_algo(self, algo):
        if self.callback:
            self.callback(algo)

class SettingsWidget(QWidget):
    def __init__(self, on_change, on_confirm):
        super().__init__()
        self.on_change = on_change
        self.on_confirm = on_confirm
        self.layout = QVBoxLayout()
        self.init_ui()

    def init_ui(self):
        label = QLabel("Settings - PyQt")
        f = label.font()
        f.setPointSize(20)
        label.setFont(f)
        label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label)
        self.rand_checkbox = QCheckBox("Use Random Map")
        self.rand_checkbox.setChecked(False)
        self.rand_checkbox.stateChanged.connect(self.on_any_change)
        self.layout.addWidget(self.rand_checkbox)
        eframe = QHBoxLayout()
        elabel = QLabel("Energy Limit:")
        eframe.addWidget(elabel)
        self.energy_entry = QLineEdit()
        self.energy_entry.setText("")
        self.energy_entry.textChanged.connect(self.on_any_change)
        eframe.addWidget(self.energy_entry)
        self.layout.addLayout(eframe)
        tframe = QHBoxLayout()
        tlabel = QLabel("Time Limit:")
        tframe.addWidget(tlabel)
        self.time_entry = QLineEdit()
        self.time_entry.setText("")
        self.time_entry.textChanged.connect(self.on_any_change)
        tframe.addWidget(self.time_entry)
        self.layout.addLayout(tframe)
        confirm_btn = QPushButton("Confirm")
        confirm_btn.setFixedSize(120,40)
        confirm_btn.clicked.connect(self.confirm_settings)
        self.layout.addWidget(confirm_btn,0,Qt.AlignCenter)
        self.setLayout(self.layout)

    def on_any_change(self):
        use_rand = self.rand_checkbox.isChecked()
        energy_val = self.energy_entry.text().strip()
        time_val = self.time_entry.text().strip()
        if self.on_change:
            self.on_change(use_rand, energy_val, time_val)

    def confirm_settings(self):
        if self.on_confirm:
            self.on_confirm()

class TrainWidget(QWidget):
    def __init__(self, tile_size, metrics, selected_metric, pause_callback, metric_callback):
        super().__init__()
        self.tile_size = tile_size
        self.metrics = metrics
        self.selected_metric = selected_metric
        self.pause = False
        self.pause_callback = pause_callback
        self.metric_callback = metric_callback
        self.environment = None
        self.loss_val = 0.0
        self.reward_val = 0.0
        self.eps_val = 0.0
        self.metric_data = []
        self.metric_data2 = []
        self.multi_curve=False
        self.smooth_sprites=[]
        self.setMouseTracking(True)
        self.load_textures()
        self.resize(self.width(), self.height())

    def attach_environment(self, env):
        self.environment=env
        self.init_smooth()

    def init_smooth(self):
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

    def load_textures(self):
        self.img_floor=None
        self.img_obstacle=None
        self.img_dirt=None
        self.img_agents=[]
        try:
            self.img_floor=QPixmap("../aset/floor.png")
        except:
            pass
        try:
            self.img_obstacle=QPixmap("../aset/stone.png")
        except:
            pass
        try:
            self.img_dirt=QPixmap("../aset/dirt.png")
        except:
            pass
        for i in range(4):
            nm=f"robot{i}.png"
            try:
                pm=QPixmap(nm)
                self.img_agents.append(pm)
            except:
                pass
        if not self.img_agents:
            self.img_agents=[None]

    def set_env_data(self, env, loss, reward, epsilon):
        self.environment=env
        self.loss_val=loss
        self.reward_val=reward
        self.eps_val=epsilon
        val=0.0
        if self.selected_metric=="Loss":
            val=self.loss_val
        elif self.selected_metric=="Reward":
            val=self.reward_val
        else:
            val=self.eps_val
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
        self.update_smooth_positions()

    def update_smooth_positions(self):
        if not self.environment:
            return
        if not hasattr(self.environment,"agent_positions"):
            return
        n=len(self.environment.agent_positions)
        for i in range(n):
            if i<len(self.smooth_sprites):
                r,c=self.environment.agent_positions[i]
                px,py=self.grid_to_px(r,c)
                self.smooth_sprites[i].set_position(px,py)
        for s in self.smooth_sprites:
            s.update()
        self.update()

    def grid_to_px(self, r,c):
        ox=50
        oy=50
        return (ox+c*self.tile_size, oy+r*self.tile_size)

    def paintEvent(self, event):
        painter=QPainter(self)
        painter.fillRect(self.rect(),QColor(30,30,30))
        if not self.environment:
            return
        self.draw_world(painter)
        self.draw_info(painter)
        self.draw_metric_graph(painter)

    def draw_world(self, painter):
        offset_x=50
        offset_y=50
        for r in range(self.environment.rows):
            for c in range(self.environment.cols):
                x=offset_x+c*self.tile_size
                y=offset_y+r*self.tile_size
                if (r,c) in self.environment.obstacles:
                    if self.img_obstacle:
                        painter.drawPixmap(x,y,self.tile_size,self.tile_size,self.img_obstacle)
                    else:
                        painter.fillRect(x,y,self.tile_size,self.tile_size,QColor(100,100,100))
                elif (r,c) in self.environment.dirts:
                    if self.img_dirt:
                        painter.drawPixmap(x,y,self.tile_size,self.tile_size,self.img_dirt)
                    else:
                        painter.fillRect(x,y,self.tile_size,self.tile_size,QColor(139,69,19))
                else:
                    if self.img_floor:
                        painter.drawPixmap(x,y,self.tile_size,self.tile_size,self.img_floor)
                    else:
                        painter.fillRect(x,y,self.tile_size,self.tile_size,QColor(200,200,200))
                pen=QPen(QColor(10,10,10))
                painter.setPen(pen)
                painter.drawRect(x,y,self.tile_size,self.tile_size)
        for i,s in enumerate(self.smooth_sprites):
            idx=i if i<len(self.img_agents) else 0
            if self.img_agents[idx]:
                painter.drawPixmap(int(s.draw_x),int(s.draw_y), self.tile_size,self.tile_size, self.img_agents[idx])
            else:
                painter.fillRect(int(s.draw_x),int(s.draw_y), self.tile_size,self.tile_size,QColor(30,144,255))

    def draw_info(self, painter):
        painter.setPen(QColor(255,255,0))
        painter.setFont(QFont("Arial",16))
        ybase=30
        painter.drawText(10,ybase,f"Loss: {round(self.loss_val,3)}")
        ybase+=25
        painter.drawText(10,ybase,f"Reward: {round(self.reward_val,3)}")
        ybase+=25
        painter.drawText(10,ybase,f"Epsilon: {round(self.eps_val,3)}")
        ybase+=25
        if self.pause:
            painter.setPen(QColor(255,0,0))
            painter.drawText(10,ybase,"[PAUSED]")
            ybase+=25

    def draw_metric_graph(self, painter):
        if len(self.metric_data)<2:
            return
        graph_left=self.width()//2
        graph_top=self.height()-260
        graph_width=self.width()//2-20
        graph_height=250
        painter.fillRect(graph_left,graph_top,graph_width,graph_height,QColor(60,60,60))
        vals=self.metric_data[-100:]
        mx=max(vals) if vals else 1
        if mx<=0:
            mx=1
        step=(graph_width-40)/(len(vals)-1)
        scale=(graph_height-40)/mx
        px=graph_left+30
        py=graph_top+graph_height-30-vals[0]*scale
        pen=QPen(QColor(0,255,0),2)
        painter.setPen(pen)
        for i in range(1,len(vals)):
            nx=graph_left+30+i*step
            ny=graph_top+graph_height-30-vals[i]*scale
            painter.drawLine(px,py,nx,ny)
            px,py=nx,ny
        if self.multi_curve and len(self.metric_data2)>1:
            vals2=self.metric_data2[-100:]
            mx2=max(vals2) if vals2 else 1
            if mx2<=0:
                mx2=1
            step2=(graph_width-40)/(len(vals2)-1)
            scale2=(graph_height-40)/mx2
            pen2=QPen(QColor(255,0,0),2)
            painter.setPen(pen2)
            px2=graph_left+30
            py2=graph_top+graph_height-30-vals2[0]*scale2
            for i in range(1,len(vals2)):
                nx2=graph_left+30+i*step2
                ny2=graph_top+graph_height-30-vals2[i]*scale2
                painter.drawLine(px2,py2,nx2,ny2)
                px2,py2=nx2,ny2
        painter.setPen(QPen(QColor(200,200,200),1))
        painter.drawLine(graph_left+30,graph_top+10, graph_left+30, graph_top+graph_height-30)
        painter.drawLine(graph_left+30,graph_top+graph_height-30, graph_left+graph_width-10, graph_top+graph_height-30)
        lb1=QFont("Arial",12)
        painter.setFont(lb1)
        painter.drawText(graph_left+5,graph_top+graph_height-25,"0")
        painter.drawText(graph_left+5,graph_top+20,str(int(mx)))
        painter.drawText(graph_left+graph_width-60,graph_top+graph_height-10,"Steps")

    def mousePressEvent(self, event):
        if event.button()==Qt.RightButton:
            mx,my=event.x(),event.y()
            gx=(mx-50)//self.tile_size
            gy=(my-50)//self.tile_size
            if self.environment and hasattr(self.environment,"mouse_edit"):
                if 0<=gy<self.environment.rows and 0<=gx<self.environment.cols:
                    self.environment.mouse_edit(gy,gx)
            self.update_smooth_positions()
    def keyPressEvent(self, event):
        if event.key()==Qt.Key_P:
            if self.pause_callback:
                self.pause_callback()
        elif event.key()==Qt.Key_M:
            if self.metric_callback:
                idx=self.metrics.index(self.selected_metric)
                idx=(idx+1)%len(self.metrics)
                new_met=self.metrics[idx]
                self.selected_metric=new_met
                self.metric_callback(new_met)
        elif event.key()==Qt.Key_C:
            self.multi_curve=not self.multi_curve
        super().keyPressEvent(event)
