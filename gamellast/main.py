import sys
import os
import json
import logging
import argparse
import time
import random
import numpy as np
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from utils.utils import ConfigLoader, ColorLogger

class UISelectionDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select UI(s)")
        self.geometry("400x240")
        self.selected_uis=[]
        self.var_pygame=tk.BooleanVar(value=False)
        self.var_tkinter=tk.BooleanVar(value=False)
        self.var_pyqt=tk.BooleanVar(value=False)
        row_frame=tk.Frame(self)
        row_frame.pack(pady=10)
        lbl=tk.Label(row_frame,text="Which UI do you want to launch?")
        lbl.pack()
        check_frame=tk.Frame(self)
        check_frame.pack(pady=10)
        c1=tk.Checkbutton(check_frame,text="PyGame",variable=self.var_pygame)
        c2=tk.Checkbutton(check_frame,text="Tkinter",variable=self.var_tkinter)
        c3=tk.Checkbutton(check_frame,text="PyQt",variable=self.var_pyqt)
        c1.pack(anchor="w")
        c2.pack(anchor="w")
        c3.pack(anchor="w")
        btn_frame=tk.Frame(self)
        btn_frame.pack(pady=10)
        confirm_btn=tk.Button(btn_frame,text="Confirm",command=self.on_confirm)
        confirm_btn.pack()
    def on_confirm(self):
        if self.var_pygame.get():
            self.selected_uis.append("pygame")
        if self.var_tkinter.get():
            self.selected_uis.append("tkinter")
        if self.var_pyqt.get():
            self.selected_uis.append("pyqt")
        if not self.selected_uis:
            messagebox.showwarning("No Selection","No UI selected, fallback to PyGame.")
            self.selected_uis=["pygame"]
        self.destroy()

class MainController:
    def __init__(self):
        self.parser=None
        self.args=None
        self.config={}
        self.logger=None
        self.init_parser()

    def init_parser(self):
        self.parser=argparse.ArgumentParser(description="RoboCleaner main.py multi-UI entry script (Extended)")
        self.parser.add_argument("--config",type=str,default="",help="Path to a JSON config file")
        self.parser.add_argument("--log_level",type=str,default="INFO",help="Logging level: DEBUG/INFO/WARNING/ERROR")
        self.parser.add_argument("--cli_passthrough",nargs="*",help="Additional args forwarded to run_game")
        self.parser.add_argument("--override",action="append",help="Override config in key=value form, can use multiple times")
        self.parser.add_argument("--no_dialog",action="store_true",help="Skip UI selection dialog and rely on config/override")

    def load_config(self, config_path):
        if not os.path.isfile(config_path):
            return
        loader=ConfigLoader(config_path)
        self.config=loader.config_dict

    def parse_overrides(self,overrides):
        if not overrides:
            return
        for ov in overrides:
            if "=" not in ov:
                continue
            key,val=ov.split("=",1)
            self.config[key.strip()]=val.strip()

    def setup_logging(self,level):
        lvl=logging.DEBUG if level.upper()=="DEBUG" else \
             logging.INFO if level.upper()=="INFO" else \
             logging.WARNING if level.upper()=="WARNING" else \
             logging.ERROR
        self.logger=ColorLogger("RoboCleanerMain")
        self.logger.setLevel(lvl)

    def build_base_command(self):
        cmd=[sys.executable,"run_game.py"]
        if "algo" in self.config:
            cmd+=["--algo",self.config["algo"]]
        if "episodes" in self.config:
            cmd+=["--episodes",str(self.config["episodes"])]
        if "fps" in self.config:
            cmd+=["--fps",str(self.config["fps"])]
        if "random_map" in self.config and self.config["random_map"] in ["true","True","1"]:
            cmd+=["--random_map"]
        if "energy" in self.config:
            cmd+=["--energy",str(self.config["energy"])]
        if "time_limit" in self.config:
            cmd+=["--time_limit",str(self.config["time_limit"])]
        if "rows" in self.config:
            cmd+=["--rows",str(self.config["rows"])]
        if "cols" in self.config:
            cmd+=["--cols",str(self.config["cols"])]
        if "obs_count" in self.config:
            cmd+=["--obs_count",str(self.config["obs_count"])]
        if "dirt_count" in self.config:
            cmd+=["--dirt_count",str(self.config["dirt_count"])]
        if "lr" in self.config:
            cmd+=["--lr",str(self.config["lr"])]
        if "seed" in self.config:
            cmd+=["--seed",str(self.config["seed"])]
        if "dueling" in self.config and self.config["dueling"] in ["true","True","1"]:
            cmd+=["--dueling"]
        if "double_dqn" in self.config and self.config["double_dqn"] in ["true","True","1"]:
            cmd+=["--double_dqn"]
        if "n_step" in self.config:
            cmd+=["--n_step",str(self.config["n_step"])]
        if "no_prioritized" in self.config and self.config["no_prioritized"] in ["true","True","1"]:
            cmd+=["--no_prioritized"]
        if "gae_lambda" in self.config:
            cmd+=["--gae_lambda",str(self.config["gae_lambda"])]
        if "eps_clip" in self.config:
            cmd+=["--eps_clip",str(self.config["eps_clip"])]
        if "k_epochs" in self.config:
            cmd+=["--k_epochs",str(self.config["k_epochs"])]
        if "batch_size" in self.config:
            cmd+=["--batch_size",str(self.config["batch_size"])]
        if "ppo_device" in self.config:
            cmd+=["--ppo_device",self.config["ppo_device"]]
        return cmd

    def parse_cli_passthrough(self,items):
        extra=[]
        i=0
        while i<len(items):
            if "--" in items[i]:
                key=items[i].replace("--","")
                if i+1<len(items):
                    val=items[i+1]
                    if "--" in val:
                        extra.append("--"+key)
                        i+=1
                    else:
                        extra.append("--"+key)
                        extra.append(val)
                        i+=2
                else:
                    extra.append("--"+key)
                    i+=1
            else:
                i+=1
        return extra

    def run_multiple_ui(self,ui_list):
        base_cmd=self.build_base_command()
        procs=[]
        for ui_name in ui_list:
            ui_cmd=list(base_cmd)
            ui_cmd+=["--ui",ui_name.strip()]
            p=subprocess.Popen(ui_cmd)
            procs.append(p)
            self.logger.info("Launched UI process for: {}".format(ui_name))
        self.logger.info("All UI processes launched (non-blocking).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Terminating child processes...")
            for pr in procs:
                pr.terminate()
            self.logger.info("All child processes terminated.")

    def run_single_ui(self,ui_name):
        base_cmd=self.build_base_command()
        base_cmd+=["--ui",ui_name]
        p=subprocess.Popen(base_cmd)
        self.logger.info("Launched single UI process: {}".format(ui_name))
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Terminating child process...")
            p.terminate()
            self.logger.info("Child process terminated.")

    def launch_dialog_and_set_ui(self):
        root=tk.Tk()
        root.withdraw()
        dialog=UISelectionDialog(root)
        dialog.grab_set()
        dialog.wait_window()
        selected=dialog.selected_uis
        root.destroy()
        if selected:
            self.config["ui"]=",".join(selected)
        else:
            self.config["ui"]="pygame"

    def execute(self):
        self.args=self.parser.parse_args()
        script_dir=os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        if self.args.config:
            self.load_config(self.args.config)
        self.parse_overrides(self.args.override)
        if self.args.cli_passthrough:
            extras=self.parse_cli_passthrough(self.args.cli_passthrough)
            for i in range(0,len(extras),2):
                if i+1<len(extras):
                    key=extras[i].replace("--","")
                    val=extras[i+1]
                    self.config[key]=val
                else:
                    key=extras[i].replace("--","")
                    self.config[key]=True
        if "log_level" not in self.config:
            self.config["log_level"]=self.args.log_level
        self.setup_logging(self.config["log_level"])
        if not self.args.no_dialog:
            if "ui" not in self.config or not self.config["ui"]:
                self.launch_dialog_and_set_ui()
        if "ui" not in self.config or not self.config["ui"]:
            self.config["ui"]="pygame"
        ui_value=self.config["ui"]
        if "," in ui_value:
            ui_list=ui_value.split(",")
            self.run_multiple_ui(ui_list)
        else:
            self.run_single_ui(ui_value)

def main():
    controller=MainController()
    controller.execute()

if __name__=="__main__":
    main()
