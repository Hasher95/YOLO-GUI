import sys

# -------------------------------------------------------------------------
# CRITICAL FIX FOR PYINSTALLER --WINDOWED MODE
# Defines stdout/stderr immediately to prevent NoneType errors on imports
# -------------------------------------------------------------------------
class DummyStream:
    def write(self, msg): pass
    def flush(self): pass

if sys.stdout is None:
    sys.stdout = DummyStream()
if sys.stderr is None:
    sys.stderr = DummyStream()

import os
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import queue  # Import queue for thread-safe communication
import logging # Needed to reset handlers
import re      # Needed for ANSI escape code stripping
import customtkinter as ctk
from ultralytics import YOLO
import shutil
import random
import yaml
import json
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import time # For timestamps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageTk # ImageTk for non-ctk if needed, but CTkImage is better
import winsound # For notifications

# -------------------------------------------------------------------------
# COMPILATION INSTRUCTIONS (PyInstaller)
# -------------------------------------------------------------------------
# To compile this script into a standalone .exe file, install PyInstaller:
# pip install pyinstaller
#
# Then run the following command in your terminal:
# pyinstaller --noconfirm --onedir --windowed --name "YoloTrainer" --collect-all ultralytics train_yolo_gui.py
#
# Notes:
# - '--windowed': Hides the console window (GUI only).
# - '--onedir': Creates a folder instead of a single file (FASTER startup).
# - Ultralytics might need hidden imports if errors occur. Using:
#   --collect-all ultralytics
# -------------------------------------------------------------------------
# TOOLTIP CLASS
# -------------------------------------------------------------------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        
        # Position below widget
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True) # Remove borders
        tw.wm_geometry(f"+{x}+{y}")
        
        # Styled like a dark tooltip
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#2B2B2B", foreground="#FFFFFF", 
                         relief='solid', borderwidth=1,
                         font=("Consolas", 9), padx=10, pady=5)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ThreadSafeRedirector(object):
    """
    Redirects stdout and stderr to a thread-safe Queue.
    """
    def __init__(self, msg_queue, tag="stdout"):
        self.msg_queue = msg_queue
        self.tag = tag

    def write(self, msg):
        # We only care about non-empty strings
        if msg:
            self.msg_queue.put(msg)

    def flush(self):
        pass

class YoloTrainerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Thread-safe logging queue
        self.log_queue = queue.Queue()
        
        # Setup redirection immediately
        # This replaces the DummyStream (or original) with our queue redirector
        self.redirector = ThreadSafeRedirector(self.log_queue, "stdout")
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        # Fix Ultralytics logging handlers to use our new stream
        self.fix_ultralytics_logging()

        # Window Setup
        self.title("YOLOv8/v11 Training GUI")
        self.geometry("950x700") 
        self.minsize(800, 500)

        # Layout Configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Tabs
        self.tabview = ctk.CTkTabview(self, width=900, height=600)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        self.tab_training = self.tabview.add("Training")
        self.tab_live = self.tabview.add("Live View")
        self.tab_prep = self.tabview.add("Dataset Prep")
        self.tab_explorer = self.tabview.add("Explorer")
        self.tab_inference = self.tabview.add("Inference") # Feature 5

        # Initialize Tabs
        self.setup_training_tab()
        self.setup_live_view_tab()
        self.setup_prep_tab()
        self.setup_explorer_tab()
        self.setup_inference_tab()

        self.stop_training_requested = False
        
        # Track training project path
        self.current_project_path = None

        # Load saved settings
        self.load_settings()

        # Start checking the log queue
        self.check_log_queue()

        # Debug Print to verify console is working
        print("GUI Started! Console redirection is active.")
        
        # Save settings on window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_config_path(self):
        """Returns the path to config.json next to the .exe or script."""
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "config.json")

    def save_settings(self):
        """Saves current GUI values to config.json."""
        try:
            settings = {
                "model": self.model_var.get(),
                "epochs": self.epochs_entry.get(),
                "batch": self.batch_entry.get(),
                "imgsz": self.imgsz_entry.get(),
                "use_gpu": self.use_gpu_var.get(),
                # Advanced
                "resume": self.resume_var.get(),
                "optimizer": self.optimizer_var.get(),
                "mosaic": self.mosaic_var.get(),
                # Paths
                "dataset_path": self.dataset_entry.get(),
                # Prep
                "prep_source": self.source_entry.get(),
                "prep_ratio": self.ratio_slider.get(),
                "prep_classes": self.classes_entry.get(),
                # Phase 4
                "lr0": self.lr0_entry.get(),
                "momentum": self.mom_entry.get(),
                "conf": self.conf_slider.get(),
                "iou": self.iou_slider.get()
            }
            with open(self.get_config_path(), "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        """Loads values from config.json into the GUI."""
        path = self.get_config_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    s = json.load(f)
                
                # Training Tab
                if "model" in s: self.model_var.set(s["model"])
                if "epochs" in s: self.epochs_entry.delete(0, "end"); self.epochs_entry.insert(0, s["epochs"])
                if "batch" in s: self.batch_entry.delete(0, "end"); self.batch_entry.insert(0, s["batch"])
                if "imgsz" in s: self.imgsz_entry.delete(0, "end"); self.imgsz_entry.insert(0, s["imgsz"])
                if "use_gpu" in s: self.use_gpu_var.set(s["use_gpu"])
                
                # Advanced
                if "resume" in s: self.resume_var.set(s["resume"])
                if "optimizer" in s: self.optimizer_var.set(s["optimizer"])
                if "mosaic" in s: self.mosaic_var.set(s["mosaic"])
                
                # Phase 4
                if "lr0" in s: self.lr0_entry.delete(0, "end"); self.lr0_entry.insert(0, s["lr0"])
                if "momentum" in s: self.mom_entry.delete(0, "end"); self.mom_entry.insert(0, s["momentum"])
                if "conf" in s: self.conf_slider.set(s["conf"])
                if "iou" in s: self.iou_slider.set(s["iou"])
                
                if "dataset_path" in s: self.dataset_entry.delete(0, "end"); self.dataset_entry.insert(0, s["dataset_path"])
                
                # Prep Tab
                if "prep_source" in s: self.source_entry.delete(0, "end"); self.source_entry.insert(0, s["prep_source"])
                if "prep_ratio" in s: 
                    self.ratio_slider.set(s["prep_ratio"])
                    self.ratio_val_label.configure(text=f"{int(s['prep_ratio']*100)}%")
                if "prep_classes" in s: self.classes_entry.delete(0, "end"); self.classes_entry.insert(0, s["prep_classes"])
            except Exception as e:
                print(f"Error loading settings: {e}")

    def on_closing(self):
        """Called when window is closed."""
        self.save_settings()
        self.destroy()

    def fix_ultralytics_logging(self):
        """
        Force Ultralytics to use our thread-safe stdout instead of the 
        default stream it might have captured (which is None in Windowed mode).
        """
        try:
            logger = logging.getLogger("ultralytics")
            # Remove all existing handlers that might hold bad streams
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            
            # Create a new handler using OUR current sys.stdout (ThreadSafeRedirector)
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s")) # Simple format
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        except Exception as e:
            print(f"Error fixing logging: {e}")

    def check_log_queue(self):
        """
        Polls the log queue every 100ms and updates the GUI with handling for \\r (progress bars).
        """
        entries = []
        try:
            # Process up to 100 messages at once
            for _ in range(100):
                entries.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass

        if entries:
            try:
                self.log_textbox.configure(state="normal")
                
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                for msg in entries:
                    msg = ansi_escape.sub('', msg)
                    
                    if not msg: continue

                    # Split by \r to handle progress bar overwrites
                    # "Loading... 10%\rLoading... 20%" -> ["Loading... 10%", "Loading... 20%"]
                    parts = msg.split('\r')
                    
                    # If msg didn't start with \r, the first part is a normal append
                    if parts[0]:
                        self.log_textbox.insert("end", parts[0])
                    
                    # For subsequent parts (or if msg started with \r), we overwrite the last line
                    for i in range(1, len(parts)):
                        # If the part is empty (e.g. "text\r\n"), skip or handle newline
                        # But typically \r is followed by text.
                        # We delete the current last line content (after the last newline)
                        # "end-1c linestart" to "end-1c" covers the text on the last line.
                        
                        # Only overwrite if there is content to write
                        if parts[i]:
                            # Delete last line's text
                            self.log_textbox.delete("end-1c linestart", "end-1c")
                            self.log_textbox.insert("end", parts[i])

                # truncate if too long (keep last 2000 lines)
                num_lines = int(self.log_textbox.index('end-1c').split('.')[0])
                if num_lines > 2000:
                    self.log_textbox.delete("1.0", "end-2000l")
                
                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")
            except Exception: pass
        
        # Schedule next check
        self.after(100, self.check_log_queue)

    def setup_training_tab(self):
        # Configure Grid for Training Tab
        self.tab_training.grid_columnconfigure(1, weight=1)
        self.tab_training.grid_rowconfigure(0, weight=1)

        # Sidebar (Settings) - Inside Tab
        self.sidebar_frame = ctk.CTkFrame(self.tab_training, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.sidebar_frame.grid_rowconfigure(9, weight=1) # Spacer at bottom

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="YOLO Trainer", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Model Selection
        self.model_label = ctk.CTkLabel(self.sidebar_frame, text="Base Model:", anchor="w")
        self.model_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.model_var = ctk.StringVar(value="yolov8n.pt")
        self.model_combobox = ctk.CTkComboBox(self.sidebar_frame,
                                            values=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", 
                                                    "yolov8l.pt", "yolov8x.pt", 
                                                    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                                            variable=self.model_var)
        self.model_combobox.grid(row=2, column=0, padx=20, pady=(0, 10))

        # Epochs
        self.epochs_label = ctk.CTkLabel(self.sidebar_frame, text="Epochs:", anchor="w")
        self.epochs_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.epochs_entry = ctk.CTkEntry(self.sidebar_frame)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=4, column=0, padx=20, pady=(0, 10))

        # Batch Size
        self.batch_label = ctk.CTkLabel(self.sidebar_frame, text="Batch Size (-1 = auto):", anchor="w")
        self.batch_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.batch_entry = ctk.CTkEntry(self.sidebar_frame)
        self.batch_entry.insert(0, "16")
        self.batch_entry.grid(row=6, column=0, padx=20, pady=(0, 10))
        
        # Image Size
        self.imgsz_label = ctk.CTkLabel(self.sidebar_frame, text="Image Size:", anchor="w")
        self.imgsz_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="nw") 
        self.imgsz_entry = ctk.CTkEntry(self.sidebar_frame)
        self.imgsz_entry.insert(0, "640")
        self.imgsz_entry.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="n")

        # GPU Option
        self.use_gpu_var = ctk.BooleanVar(value=True)
        self.gpu_switch = ctk.CTkCheckBox(self.sidebar_frame, text="Use GPU (device=0)", variable=self.use_gpu_var)
        self.gpu_switch.grid(row=9, column=0, padx=20, pady=(10, 10), sticky="n")

        # --- Advanced Options ---
        self.advanced_label = ctk.CTkLabel(self.sidebar_frame, text="Advanced:", anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        self.advanced_label.grid(row=10, column=0, padx=20, pady=(10, 0), sticky="w")

        # Resume
        self.resume_var = ctk.BooleanVar(value=False)
        self.resume_switch = ctk.CTkCheckBox(self.sidebar_frame, text="Resume Training", variable=self.resume_var)
        self.resume_switch.grid(row=11, column=0, padx=20, pady=(5, 0), sticky="w")
        
        # Mosaic
        self.mosaic_var = ctk.BooleanVar(value=True)
        self.mosaic_switch = ctk.CTkCheckBox(self.sidebar_frame, text="Use Mosaic Aug", variable=self.mosaic_var)
        self.mosaic_switch.grid(row=12, column=0, padx=20, pady=(5, 0), sticky="w")
        ToolTip(self.mosaic_switch, "Mosaic augmentation combines 4 images into 1.\nHighly effective for small objects.\n[Default: Checked] [Rec: Checked]")

        # Optimizer
        self.optim_label = ctk.CTkLabel(self.sidebar_frame, text="Optimizer:", anchor="w")
        self.optim_label.grid(row=13, column=0, padx=20, pady=(5, 0), sticky="w")
        self.optimizer_var = ctk.StringVar(value="auto")
        self.optimizer_menu = ctk.CTkOptionMenu(self.sidebar_frame, variable=self.optimizer_var,
                                                values=["auto", "SGD", "Adam", "AdamW"])
        self.optimizer_menu.grid(row=14, column=0, padx=20, pady=(0, 10))
        ToolTip(self.optimizer_menu, "Algorithm to update weights.\n'auto' works best for most users.\n[Default: auto] [Rec: AdamW for complex datasets]")

        # LR0
        self.lr0_label = ctk.CTkLabel(self.sidebar_frame, text="Learning Rate (lr0):", anchor="w")
        self.lr0_label.grid(row=15, column=0, padx=20, pady=(5, 0), sticky="w")
        self.lr0_entry = ctk.CTkEntry(self.sidebar_frame)
        self.lr0_entry.insert(0, "0.01")
        self.lr0_entry.grid(row=16, column=0, padx=20, pady=(0, 10))
        ToolTip(self.lr0_entry, "Initial Learning Rate.\nLower values are stabler; higher learn faster.\n[Default: 0.01] [Rec: 0.01 std / 0.001 fine-tune]")

        # Momentum
        self.mom_label = ctk.CTkLabel(self.sidebar_frame, text="Momentum:", anchor="w")
        self.mom_label.grid(row=17, column=0, padx=20, pady=(5, 0), sticky="w")
        self.mom_entry = ctk.CTkEntry(self.sidebar_frame)
        self.mom_entry.insert(0, "0.937")
        self.mom_entry.grid(row=18, column=0, padx=20, pady=(0, 10))
        ToolTip(self.mom_entry, "Accelerates gradient descent and dampens oscillations.\n[Default: 0.937] [Rec: 0.9-0.98]")

        # --- Tools (Export) ---
        self.tools_label = ctk.CTkLabel(self.sidebar_frame, text="Tools:", anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        self.tools_label.grid(row=19, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.export_fmt_var = ctk.StringVar(value="onnx")
        self.export_menu = ctk.CTkOptionMenu(self.sidebar_frame, variable=self.export_fmt_var,
                                             values=["onnx", "torchscript", "openvino", "engine", "coreml", "tflite"])
        self.export_menu.grid(row=20, column=0, padx=20, pady=(5, 5))
        
        self.export_btn = ctk.CTkButton(self.sidebar_frame, text="Export Model", fg_color="#555555",
                                        command=self.export_model_thread)
        self.export_btn.grid(row=21, column=0, padx=20, pady=(0, 20))

        # Main Area
        self.main_train_frame = ctk.CTkFrame(self.tab_training, corner_radius=0, fg_color="transparent")
        self.main_train_frame.grid(row=0, column=1, sticky="nsew")
        self.main_train_frame.grid_rowconfigure(2, weight=1) 
        self.main_train_frame.grid_columnconfigure(0, weight=1)

        # 1. Dataset Selection
        self.dataset_frame = ctk.CTkFrame(self.main_train_frame)
        self.dataset_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.dataset_frame.grid_columnconfigure(0, weight=1)

        self.dataset_label = ctk.CTkLabel(self.dataset_frame, text="Dataset Configuration (data.yaml)", font=ctk.CTkFont(size=14, weight="bold"))
        self.dataset_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        self.dataset_entry = ctk.CTkEntry(self.dataset_frame, placeholder_text="Path to data.yaml")
        self.dataset_entry.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="ew")
        
        self.browse_button = ctk.CTkButton(self.dataset_frame, text="Browse", width=100, command=self.browse_file)
        self.browse_button.grid(row=1, column=1, padx=(5, 10), pady=10)

        # 2. Action Button
        self.start_button = ctk.CTkButton(self.main_train_frame, text="START TRAINING", 
                                          font=ctk.CTkFont(size=16, weight="bold"),
                                          height=50, fg_color="green", hover_color="darkgreen",
                                          command=self.start_training_thread)
        self.start_button.grid(row=1, column=0, sticky="ew", pady=(0, 20))

        # LOGS & GRAPHS AREA
        # ---------------------------
        # Right side split: Top for Logs, Bottom for Graphs
        self.right_frame = ctk.CTkFrame(self.main_train_frame)
        self.right_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 0), pady=0) # Adjusted grid to fit main_train_frame
        self.right_frame.grid_rowconfigure(1, weight=1) # Logs
        self.right_frame.grid_rowconfigure(3, weight=1) # Graphs
        self.right_frame.grid_columnconfigure(0, weight=1)

        # 1. Logs (Top)
        self.log_label = ctk.CTkLabel(self.right_frame, text="Console Output:", anchor="w", font=ctk.CTkFont(weight="bold"))
        self.log_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

        self.log_textbox = ctk.CTkTextbox(self.right_frame, height=250, activate_scrollbars=True)
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        # Configure Log Aesthetics
        self.log_textbox.configure(font=("Consolas", 11), fg_color="#1E1E1E", text_color="#D4D4D4")
        self.log_textbox.tag_config("linestart", foreground="#D4D4D4") 

        # 2. Graphs (Bottom)
        self.graph_frame = ctk.CTkFrame(self.right_frame)
        self.graph_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.graph_frame.grid_columnconfigure(0, weight=1)
        self.graph_frame.grid_rowconfigure(0, weight=1)
        
        # Matplotlib Init
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_facecolor('#2B2B2B') # Match dark theme roughly
        self.ax1 = self.figure.add_subplot(121)
        self.ax2 = self.figure.add_subplot(122)
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=0.4)

        # Set axes background and text color for dark theme
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#2B2B2B')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        # Start Graph Loop
        self.after(10000, self.update_graphs) # 10s delay start

    def setup_live_view_tab(self):
        """Feature 2: Live view of validation images during training."""
        self.tab_live.grid_columnconfigure(0, weight=1)
        self.tab_live.grid_rowconfigure(0, weight=1)
        
        # Scrollable frame for the image
        self.live_scroll = ctk.CTkScrollableFrame(self.tab_live, label_text="Live Training Monitor")
        self.live_scroll.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.live_image_label = ctk.CTkLabel(self.live_scroll, text="Waiting for validation images...\n(These appear after the first few training steps/epochs)")
        self.live_image_label.pack(expand=True, fill="both", padx=10, pady=20)
        
        # Initialize internal state
        self.current_live_image_path = None
        
        # Start auto-refresh loop
        self.after(5000, self.update_live_view)

    def update_live_view(self):
        """Polls for images in the active training directory and cycles through them."""
        # 1. Active Directory Discovery (Keep discovering newer 'trainX' folders during run)
        is_active = hasattr(self, 'train_thread') and self.train_thread and self.train_thread.is_alive()
        
        if is_active:
            project_path = getattr(self, 'current_project_path', None)
            if project_path and os.path.exists(project_path):
                try:
                    subdirs = [os.path.join(project_path, d) for d in os.listdir(project_path) 
                                if os.path.isdir(os.path.join(project_path, d)) and d.startswith("train")]
                    if subdirs:
                        latest = max(subdirs, key=os.path.getmtime)
                        if not hasattr(self, 'current_train_dir') or self.current_train_dir != latest:
                            self.current_train_dir = latest
                            print(f"[LiveView] Switched to latest training directory: {latest}")
                except:
                    pass

        # 2. File Selection & Cycling
        if hasattr(self, 'current_train_dir') and self.current_train_dir and os.path.exists(self.current_train_dir):
            try:
                files = os.listdir(self.current_train_dir)
                img_exts = (".jpg", ".jpeg", ".png")
                candidates = [f for f in files if f.lower().endswith(img_exts)]
                
                # Categories with priority
                preds = sorted([f for f in candidates if "val_batch" in f and "_pred" in f])
                results = [f for f in candidates if "results" in f and f.endswith(".png")]
                labels = sorted([f for f in candidates if "val_batch" in f and "_labels" in f])
                trains = sorted([f for f in candidates if "train_batch" in f])
                
                # Determine Pool
                pool = []
                if preds:
                    pool = preds + results # Cycle between predictions and the graph
                elif results:
                    pool = results + labels
                elif labels:
                    pool = labels
                else:
                    pool = trains
                
                if pool:
                    if not hasattr(self, '_live_idx'): self._live_idx = 0
                    self._live_idx = (self._live_idx + 1) % len(pool)
                    target = pool[self._live_idx]
                    
                    target_path = os.path.join(self.current_train_dir, target)
                    self.live_scroll.configure(label_text=f"Live Monitor: {target} (Auto-refreshing 5s)")
                    
                    pil_img = Image.open(target_path)
                    # Scale down for display
                    w_base = 800
                    w_percent = (w_base / float(pil_img.size[0]))
                    h_size = int((float(pil_img.size[1]) * float(w_percent)))
                    pil_img = pil_img.resize((w_base, h_size), Image.Resampling.LANCZOS)
                    
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w_base, h_size))
                    self.live_image_label.configure(image=ctk_img, text="")
                    self.live_image_label.image = ctk_img # Keep ref

            except Exception as e:
                pass # Silent to avoid flickering/spam

        # Schedule next check (faster for cycling)
        self.after(5000, self.update_live_view)

    def setup_prep_tab(self):
        self.tab_prep.grid_columnconfigure(0, weight=1)

        # Step 1: Source
        self.prep_label = ctk.CTkLabel(self.tab_prep, text="Dataset Preparation Tool", font=ctk.CTkFont(size=18, weight="bold"))
        self.prep_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.source_frame = ctk.CTkFrame(self.tab_prep)
        self.source_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.source_frame.grid_columnconfigure(0, weight=1)

        self.source_label = ctk.CTkLabel(self.source_frame, text="1. Source Folder (Mixed images & txts):")
        self.source_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.source_entry = ctk.CTkEntry(self.source_frame, placeholder_text="Select folder with raw images and labels")
        self.source_entry.grid(row=1, column=0, padx=(10,5), pady=(0,10), sticky="ew")
        
        self.browse_source_btn = ctk.CTkButton(self.source_frame, text="Browse", width=100, command=self.browse_dataset)
        self.browse_source_btn.grid(row=1, column=1, padx=(5,10), pady=(0,10))

        # Step 2: Config
        self.config_frame = ctk.CTkFrame(self.tab_prep)
        self.config_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.config_frame.grid_columnconfigure(0, weight=1)

        self.ratio_label = ctk.CTkLabel(self.config_frame, text="2. Split Ratio (Train size):")
        self.ratio_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.ratio_slider = ctk.CTkSlider(self.config_frame, from_=0.5, to=0.99, number_of_steps=50)
        self.ratio_slider.set(0.8)
        self.ratio_slider.grid(row=1, column=0, padx=10, pady=(0,5), sticky="ew")
        self.ratio_val_label = ctk.CTkLabel(self.config_frame, text="80%")
        self.ratio_val_label.grid(row=1, column=1, padx=10, sticky="w")
        
        self.ratio_slider.configure(command=lambda val: self.ratio_val_label.configure(text=f"{int(val*100)}%"))

        self.classes_label = ctk.CTkLabel(self.config_frame, text="3. Class Names (comma separated):")
        self.classes_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.classes_entry = ctk.CTkEntry(self.config_frame, placeholder_text="e.g. cat, dog, person")
        self.classes_entry.grid(row=3, column=0, columnspan=2, padx=10, pady=(0,10), sticky="ew")

        # Step 3: Action
        self.process_btn = ctk.CTkButton(self.tab_prep, text="GENERATE DATASET & YAML", 
                                       font=ctk.CTkFont(size=16, weight="bold"),
                                       height=50, fg_color="#D35400", hover_color="#A04000",
                                       command=self.start_processing_thread)
        self.process_btn.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        # Log for Prep
        self.prep_log_textbox = ctk.CTkTextbox(self.tab_prep, height=150)
        self.prep_log_textbox.grid(row=4, column=0, padx=20, pady=(0,10), sticky="nsew")
        self.tab_prep.grid_rowconfigure(4, weight=1)

    def update_graphs(self):
        """Reads results.csv and updates matplotlib graphs."""
        if not self.tabview.get() == "Training": 
            # Schedule check regardless, or pause? Better schedule.
            self.after(10000, self.update_graphs)
            return

        # Optimization: Only update if actively training (project path set)
        if not self.current_project_path or not os.path.exists(self.current_project_path):
             self.after(10000, self.update_graphs)
             return
            
        try:
            # Usually results.csv is in project/name/results.csv
            # We defined project=.../training_runs and name="train"
            # Ultralytics often creates train, train2, train3...
            # We need to find the latest 'train*' folder in 'training_runs'
            results_file = None
            
            if os.path.exists(self.current_project_path):
                 # Simple check: assumes we are looking at the specific run folders inside
                 # Actually, model.train(project=P, name=N) -> output is P/N
                 # But if N exists, it becomes N2.
                 # Let's search inside self.current_project_path (which is .../training_runs) containing 'train*'
                
                # We need the ACTUAL run directory. 
                # Let's assume the latest modified folder in training_runs is the current one
                subdirs = [os.path.join(self.current_project_path, d) for d in os.listdir(self.current_project_path) if os.path.isdir(os.path.join(self.current_project_path, d)) and "train" in d]
                if subdirs:
                    latest_subdir = max(subdirs, key=os.path.getmtime)
                    candidate = os.path.join(latest_subdir, "results.csv")
                    if os.path.exists(candidate):
                        results_file = candidate

            if results_file:
                df = pd.read_csv(results_file)
                # Strip spaces from column names
                df.columns = [c.strip() for c in df.columns]
                
                # Plot 1: Loss
                self.ax1.clear()
                self.ax1.set_title("Box Loss", color="white", fontsize=8)
                self.ax1.set_xlabel("Epoch", color="white", fontsize=7)
                self.ax1.tick_params(axis='x', colors='white', labelsize=6)
                self.ax1.tick_params(axis='y', colors='white', labelsize=6)
                self.ax1.plot(df['epoch'], df['train/box_loss'], label='Train', color='#00ff00')
                if 'val/box_loss' in df.columns:
                    self.ax1.plot(df['epoch'], df['val/box_loss'], label='Val', color='#ff9900')
                self.ax1.legend(fontsize=6)
                self.ax1.grid(True, linestyle='--', alpha=0.3)
                self.ax1.set_facecolor('#2B2B2B')

                # Plot 2: mAP
                self.ax2.clear()
                self.ax2.set_title("mAP 50-95", color="white", fontsize=8)
                self.ax2.set_xlabel("Epoch", color="white", fontsize=7)
                self.ax2.tick_params(axis='x', colors='white', labelsize=6)
                self.ax2.tick_params(axis='y', colors='white', labelsize=6)
                
                metric_col = [c for c in df.columns if "mAP50-95" in c]
                if metric_col:
                    m_col = metric_col[0]
                    self.ax2.plot(df['epoch'], df[m_col], color='#00aaff')
                    
                    # Find Best Epoch
                    if not df.empty and df[m_col].max() > 0:
                        best_idx = df[m_col].idxmax()
                        best_epoch = df.loc[best_idx, 'epoch']
                        best_val = df.loc[best_idx, m_col]
                        
                        # Add Vertical Line to mAP
                        self.ax2.axvline(x=best_epoch, color='white', linestyle='--', alpha=0.6)
                        self.ax2.annotate(f"Best: {best_val:.3f}", 
                                         xy=(best_epoch, best_val),
                                         xytext=(5, 5), textcoords='offset points',
                                         color='white', fontsize=7, fontweight='bold')
                        
                        # Add Vertical Line to Loss (ax1)
                        self.ax1.axvline(x=best_epoch, color='white', linestyle='--', alpha=0.4)
                        self.ax1.text(best_epoch, self.ax1.get_ylim()[1]*0.9, "Best", 
                                     color='white', fontsize=6, ha='center', alpha=0.8)

                self.ax1.grid(True, linestyle='--', alpha=0.3)
                self.ax2.grid(True, linestyle='--', alpha=0.3)
                self.ax2.set_facecolor('#2B2B2B')

                self.canvas.draw()

        except Exception as e:
            # print(f"Graph Error: {e}") # Reduce noise
            pass
        finally:
             self.after(10000, self.update_graphs) # 10s interval

    def browse_dataset(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, folder_path)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if file_path:
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, file_path)

    def log_prep(self, message):
        # We can also use queue for prep logs, but pure tkinter widgets are fine if not too heavy
        # However, to avoid "freeze" during heavy file copy loop, it's better to queue
        # For simplicity, prep logs are separate here, but we will protect them
        self.after(0, lambda: self._safe_log_prep(message))
    
    def _safe_log_prep(self, message):
        self.prep_log_textbox.insert("end", message + "\n")
        self.prep_log_textbox.see("end")

    def start_processing_thread(self):
        # Save settings on action
        self.save_settings()
        
        self.process_btn.configure(state="disabled", text="Processing...")
        threading.Thread(target=self.run_process_dataset, daemon=True).start()

    # -------------------------------------------------------------------------
    # EXPLORER TAB (Feature 3)
    # -------------------------------------------------------------------------
    def setup_explorer_tab(self):
        self.tab_explorer.grid_columnconfigure(0, weight=1)
        self.tab_explorer.grid_rowconfigure(1, weight=1)

        # Controls
        self.exp_ctrl_frame = ctk.CTkFrame(self.tab_explorer)
        self.exp_ctrl_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        self.exp_btn = ctk.CTkButton(self.exp_ctrl_frame, text="Load Image Folder", command=self.load_explorer_images)
        self.exp_btn.pack(side="left", padx=10, pady=10)
        
        self.exp_info = ctk.CTkLabel(self.exp_ctrl_frame, text="No folder loaded.")
        self.exp_info.pack(side="left", padx=10)

        # Scrollable Grid
        self.exp_scroll = ctk.CTkScrollableFrame(self.tab_explorer, label_text="Image Grid")
        self.exp_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

    def load_explorer_images(self):
        folder = filedialog.askdirectory(title="Select folder containing images")
        if not folder:
            return
        
        self.exp_info.configure(text=f"Loading: {folder}...")
        
        # Clear previous
        for widget in self.exp_scroll.winfo_children():
            widget.destroy()

        # Find images
        exts = ('.jpg', '.png', '.jpeg', '.bmp')
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        files = files[:100] # Limit to 100 for performance
        
        row, col = 0, 0
        cols_max = 5
        
        for f in files:
            path = os.path.join(folder, f)
            try:
                # Thumbnail
                pil_img = Image.open(path)
                pil_img.thumbnail((100, 100))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(100, 100))
                
                btn = ctk.CTkButton(self.exp_scroll, text=f, image=ctk_img, compound="top",
                                    command=lambda p=path: self.show_full_image(p))
                btn.grid(row=row, column=col, padx=5, pady=5)
                
                col += 1
                if col >= cols_max:
                    col = 0
                    row += 1
            except Exception:
                pass
        
        self.exp_info.configure(text=f"Loaded {len(files)} images from {os.path.basename(folder)}")

    def show_full_image(self, img_path):
        """Shows image with bounding boxes in a new window."""
        top = ctk.CTkToplevel(self)
        top.title(os.path.basename(img_path))
        top.geometry("800x600")
        
        try:
            pil_img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            w, h = pil_img.size
            
            # Look for label file
            # Assumptions: 
            # 1. Same folder, .txt
            # 2. ../labels/filename.txt
            
            label_path = None
            basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Check 1: Same folder
            p1 = os.path.join(os.path.dirname(img_path), basename + ".txt")
            # Check 2: Top level structure (images/.. -> labels/..)
            parent = os.path.dirname(os.path.dirname(img_path)) # ../
            p2 = os.path.join(parent, "labels", os.path.basename(os.path.dirname(img_path)), basename + ".txt") # labels/train/file.txt
            p3 = os.path.join(parent, "labels", basename + ".txt") # labels/file.txt
            
            for p in [p1, p2, p3]:
                if os.path.exists(p):
                    label_path = p
                    break
            
            if label_path:
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = parts[0]
                            cx, cy, bw, bh = map(float, parts[1:5])
                            
                            # Convert to pixels
                            x1 = (cx - bw/2) * w
                            y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w
                            y2 = (cy + bh/2) * h
                            
                            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
                            draw.text((x1, y1-10), f"Class {cls}", fill="#00FF00")
            
            # Resize for display if too big
            display_h = 550
            ratio = display_h / h
            new_w = int(w * ratio)
            pil_img_resized = pil_img.resize((new_w, display_h))
            
            ctk_out = ctk.CTkImage(light_image=pil_img_resized, dark_image=pil_img_resized, size=(new_w, display_h))
            lbl = ctk.CTkLabel(top, text="", image=ctk_out)
            lbl.pack(expand=True, fill="both")
            
        except Exception as e:
            print(e)

    def run_process_dataset(self):
        try:
            source_dir = self.source_entry.get().strip().replace('"', '')
            class_names_str = self.classes_entry.get().strip()
            split_ratio = self.ratio_slider.get()

            if not os.path.exists(source_dir):
                self.log_prep("ERROR: Source directory does not exist.")
                return
            
            if not class_names_str:
                self.log_prep("ERROR: Please enter class names.")
                return
            
            class_names = [c.strip() for c in class_names_str.split(',') if c.strip()]
            
            self.log_prep(f"Scanning {source_dir}...")
            
            # Find pairs
            supported_ext = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
            files = [f for f in os.listdir(source_dir) if f.lower().endswith(supported_ext)]
            
            pairs = []
            for img_file in files:
                basename = os.path.splitext(img_file)[0]
                txt_file = basename + '.txt'
                if os.path.exists(os.path.join(source_dir, txt_file)):
                    pairs.append((img_file, txt_file))
            
            if not pairs:
                self.log_prep("ERROR: No valid image+txt pairs found.")
                return

            self.log_prep(f"Found {len(pairs)} image-label pairs.")
            
            # Create Structure
            target_root = os.path.join(source_dir, "dataset_split")
            if os.path.exists(target_root):
                 self.log_prep("WARNING: 'dataset_split' folder exists. Merging content.")
            else:
                os.makedirs(target_root)

            dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
            for d in dirs:
                os.makedirs(os.path.join(target_root, d), exist_ok=True)

            # Shuffle and Split
            random.shuffle(pairs)
            split_idx = int(len(pairs) * split_ratio)
            train_set = pairs[:split_idx]
            val_set = pairs[split_idx:]
            
            self.log_prep(f"Split: {len(train_set)} Train, {len(val_set)} Val.")

            # Copy Files
            def copy_files(file_set, type_name):
                for img, txt in file_set:
                    # Copy Image
                    shutil.copy2(os.path.join(source_dir, img), 
                                 os.path.join(target_root, f'images/{type_name}', img))
                    # Copy Label
                    shutil.copy2(os.path.join(source_dir, txt), 
                                 os.path.join(target_root, f'labels/{type_name}', txt))
            
            self.log_prep("Copying training files...")
            copy_files(train_set, 'train')
            self.log_prep("Copying validation files...")
            copy_files(val_set, 'val')

            # Generate YAML
            yaml_path = os.path.join(target_root, "data.yaml")
            data_yaml = {
                'path': os.path.abspath(target_root),
                'train': 'images/train',
                'val': 'images/val',
                'nc': len(class_names),
                'names': class_names
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False)
            
            self.log_prep(f"SUCCESS: data.yaml created at {yaml_path}")
            
            # Auto-load into trainer
            self.after(0, lambda: self.dataset_entry.delete(0, "end"))
            self.after(0, lambda: self.dataset_entry.insert(0, yaml_path))
            self.after(0, lambda: self.tabview.set("Training"))
            self.after(0, lambda: messagebox.showinfo("Done", "Dataset prepared and loaded!"))

        except Exception as e:
            self.log_prep(f"CRITICAL ERROR: {e}")
            print(e)
        finally:
             self.after(0, lambda: self.process_btn.configure(state="normal", text="GENERATE DATASET & YAML"))

    def start_training_thread(self):
        """
        Handles the logic for the Start/Stop button.
        """
        # If already training, this button acts as a "STOP" button
        if hasattr(self, "training_thread") and self.training_thread.is_alive():
            if messagebox.askyesno("Stop Training", "Are you sure you want to stop the current training session?"):
                self.stop_training_requested = True
                self.start_button.configure(state="disabled", text="Stopping...")
            return

        # Start fresh training
        data_yaml_path = self.dataset_entry.get().strip().replace('"', '').replace("'", "")
        if not data_yaml_path or not os.path.exists(data_yaml_path):
            messagebox.showerror("Error", "Please select a valid data.yaml file.")
            return

        # Get params from GUI
        try:
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_entry.get())
            imgsz = int(self.imgsz_entry.get())
            device_str = "0" if self.use_gpu_var.get() else "cpu"
            model_name = self.model_var.get()
            
            # Advanced
            resume_flag = self.resume_var.get()
            optim_choice = self.optimizer_var.get()
            use_mosaic = self.mosaic_var.get()

            # Hyperparams (Feature 3)
            lr0 = float(self.lr0_entry.get()) if self.lr0_entry.get().strip() else 0.01
            momentum = float(self.mom_entry.get()) if self.mom_entry.get().strip() else 0.937
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric values in settings.")
            return

        # Feature 4: Robustness (Check data.yaml content)
        try:
            with open(data_yaml_path, 'r') as f:
                y = yaml.safe_load(f)
                if 'train' not in y or 'val' not in y:
                    messagebox.showerror("Error", "data.yaml is missing 'train' or 'val' paths.")
                    return
        except Exception as e:
            messagebox.showerror("Error", f"Could not read data.yaml: {e}")
            return

        # Save settings automatically
        self.save_settings()

        # Prepare UI
        self.start_button.configure(text="STOP TRAINING", fg_color="red", hover_color="darkred")
        self.stop_training_requested = False

        # Launch thread
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(model_name, data_yaml_path, epochs, imgsz, batch_size, device_str, resume_flag, optim_choice, use_mosaic, lr0, momentum)
        )
        self.training_thread.start()

    def run_training(self, model_name, data_yaml_path, epochs, imgsz, batch_size, device_str, resume_flag, optim_choice, use_mosaic, lr0, momentum):
        try:
            # 1. Initialize Model
            # FORCE RE-APPLY LOGGING REDIRECTION IN THREAD
            sys.stdout = self.redirector
            sys.stderr = self.redirector
            print("[INFO] Starting Training Thread...") # Debug check

            model = YOLO(model_name)
            self.fix_ultralytics_logging() # Re-hook after YOLO init resets it

            # ---------------------------------------------------------
            # STOP & DIR CAPTURE MECHANISM (CALLBACKS)
            # ---------------------------------------------------------
            def on_train_start(trainer):
                # Capture the actual directory being used (train, train2, etc.)
                self.current_train_dir = str(trainer.save_dir)
                print(f"[LiveView] Active directory detected: {self.current_train_dir}")

            def on_train_batch_start(trainer):
                if self.stop_training_requested:
                    raise Exception("TRAINING_STOPPED_BY_USER")
            
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_batch_start", on_train_batch_start)

            # ---------------------------------------------------------
            # DETERMINE OUTPUT DIRECTORY
            # ---------------------------------------------------------
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            
            project_path = os.path.join(base_path, "training_runs")
            
            print(f"Training results will be saved to: {project_path}")
            self.current_project_path = project_path # Expose for graph updater
            
            # Temporary fallback, will be overwritten by on_train_start callback
            self.current_train_dir = os.path.join(project_path, "train")

            results = model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device_str,
                workers=8,
                project=project_path,
                name="train",
                exist_ok=True, # Force use of 'train' folder to make tracking easier
                # Advanced
                resume=resume_flag,
                optimizer=optim_choice,
                mosaic=1.0 if use_mosaic else 0.0,
                lr0=lr0,
                momentum=momentum
            )
            
            # On completion
            print(f"\n[INFO] Training Completed Successfully!\nModels saved in: {project_path}\n")
            messagebox.showinfo("Success", f"Training Finished!\nResults saved to:\n{project_path}")
            winsound.Beep(1000, 500) # Notification Sound
            
        except Exception as e:
            if str(e) == "TRAINING_STOPPED_BY_USER":
                print(f"\n[INFO] Training stopped by user.\n")
                messagebox.showwarning("Stopped", "Training was interrupted by the user.")
            else:
                print(f"\n[ERROR] Start Failed: {e}\n")
                # import traceback
                # traceback.print_exc()
                messagebox.showerror("Training Error", f"An error occurred:\n{e}")
        finally:
            self.after(0, lambda: self.start_button.configure(state="normal", text="START TRAINING", fg_color="green", hover_color="darkgreen"))
            self.stop_training_requested = False
            winsound.Beep(800, 300)

    def export_model_thread(self):
        threading.Thread(target=self.run_export, daemon=True).start()

    def run_export(self):
        model_path = self.model_var.get()
        fmt = self.export_fmt_var.get()
        
        if not model_path: 
            messagebox.showerror("Error", "No model selected.")
            return

        try:
            self.export_btn.configure(state="disabled", text="Exporting...")
            print(f"Exporting {model_path} to {fmt}...")
            model = YOLO(model_path)
            out = model.export(format=fmt)
            print(f"Export Success: {out}")
            messagebox.showinfo("Export Done", f"Model exported to {fmt}!")
        except Exception as e:
            print(f"Export Error: {e}")
            messagebox.showerror("Export Error", f"{e}")
        finally:
             self.after(0, lambda: self.export_btn.configure(state="normal", text="Export Model"))

    # -------------------------------------------------------------------------
    # INFERENCE TAB (Feature 5)
    # -------------------------------------------------------------------------
    def setup_inference_tab(self):
        self.tab_inference.grid_columnconfigure(0, weight=1)
        self.tab_inference.grid_rowconfigure(2, weight=1)
        
        # 1. Selection
        self.inf_frame = ctk.CTkFrame(self.tab_inference)
        self.inf_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Model A
        self.inf_model_a_btn = ctk.CTkButton(self.inf_frame, text="Model A (.pt)", command=lambda: self.browse_inf_model('A'))
        self.inf_model_a_btn.grid(row=0, column=0, padx=10, pady=5)
        self.inf_model_a_lbl = ctk.CTkLabel(self.inf_frame, text="yolov8n.pt")
        self.inf_model_a_lbl.grid(row=0, column=1, padx=5, sticky="w")
        self.inf_model_a_path = "yolov8n.pt"

        # Model B
        self.inf_model_b_btn = ctk.CTkButton(self.inf_frame, text="Model B (Optional)", fg_color="#555555", command=lambda: self.browse_inf_model('B'))
        self.inf_model_b_btn.grid(row=1, column=0, padx=10, pady=5)
        self.inf_model_b_lbl = ctk.CTkLabel(self.inf_frame, text="None")
        self.inf_model_b_lbl.grid(row=1, column=1, padx=5, sticky="w")
        self.inf_model_b_path = None

        # Media (Batch)
        self.inf_media_btn = ctk.CTkButton(self.inf_frame, text="Select Images/Videos (Batch)", command=self.browse_inf_media)
        self.inf_media_btn.grid(row=2, column=0, padx=10, pady=10)
        self.inf_media_lbl = ctk.CTkLabel(self.inf_frame, text="No media selected")
        self.inf_media_lbl.grid(row=2, column=1, padx=5, sticky="w")
        self.inf_media_paths = []

        # Save Checkbox
        self.save_comp_var = ctk.BooleanVar(value=False)
        self.save_comp_switch = ctk.CTkCheckBox(self.inf_frame, text="Auto-Save Comparisons", variable=self.save_comp_var)
        self.save_comp_switch.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        # Confidence & IoU
        ctk.CTkLabel(self.inf_frame, text="Confidence:").grid(row=4, column=0, padx=10, pady=(5,0), sticky="e")
        self.conf_slider = ctk.CTkSlider(self.inf_frame, from_=0.01, to=1.0, number_of_steps=100)
        self.conf_slider.set(0.25)
        self.conf_slider.grid(row=4, column=1, padx=10, pady=(5,0), sticky="ew")
        ToolTip(self.conf_slider, "Minimum probability for a detection to be kept.\n[Default: 0.25] [Rec: 0.25 - 0.5]")
        
        ctk.CTkLabel(self.inf_frame, text="IoU:").grid(row=5, column=0, padx=10, pady=(5,0), sticky="e")
        self.iou_slider = ctk.CTkSlider(self.inf_frame, from_=0.01, to=1.0, number_of_steps=100)
        self.iou_slider.set(0.7)
        self.iou_slider.grid(row=5, column=1, padx=10, pady=(5,0), sticky="ew")
        ToolTip(self.iou_slider, "Non-Maximum Suppression (NMS) threshold.\nControls overlapping box merging.\n[Default: 0.7] [Rec: 0.45 - 0.7]")

        self.inf_run_btn = ctk.CTkButton(self.inf_frame, text="RUN BATCH INFERENCE", fg_color="#D35400", command=self.run_inference)
        self.inf_run_btn.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # 2. Display
        self.inf_display = ctk.CTkScrollableFrame(self.tab_inference, label_text="Results")
        self.inf_display.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
    def browse_inf_model(self, slot):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.pt")])
        if p:
            if slot == 'A':
                self.inf_model_a_path = p
                self.inf_model_a_lbl.configure(text=os.path.basename(p))
            else:
                self.inf_model_b_path = p
                self.inf_model_b_lbl.configure(text=os.path.basename(p))
            
    def browse_inf_media(self):
        paths = filedialog.askopenfilenames(filetypes=[("Media", "*.jpg *.png *.jpeg *.mp4 *.avi *.mkv")])
        if paths:
            self.inf_media_paths = paths
            self.inf_media_lbl.configure(text=f"{len(paths)} files selected")

    def run_inference(self):
        if not self.inf_media_paths or not self.inf_model_a_path:
            messagebox.showerror("Error", "Please select Model A and at least one image/video.")
            return

        self.inf_run_btn.configure(state="disabled", text="Running Batch...")
        
        # Clear previous results
        for w in self.inf_display.winfo_children():
            w.destroy()

        def _run_batch():
            try:
                # Load models
                model_a = YOLO(self.inf_model_a_path)
                model_b = None
                if self.inf_model_b_path:
                    model_b = YOLO(self.inf_model_b_path)
                
                # Output dir for comparisons
                save_dir = None
                if self.save_comp_var.get():
                     if getattr(sys, 'frozen', False):
                        base = os.path.dirname(sys.executable)
                     else:
                        base = os.path.dirname(os.path.abspath(__file__))
                     save_dir = os.path.join(base, "comparisons")
                     os.makedirs(save_dir, exist_ok=True)

                for media_path in self.inf_media_paths:
                    # Get current slider values
                    conf_val = self.conf_slider.get()
                    iou_val = self.iou_slider.get()

                    # Run A
                    res_a = model_a.predict(source=media_path, save=False, conf=conf_val, iou=iou_val, stream=True)
                    im_a_bgr = next(res_a).plot() # Get first frame/image
                    im_a_rgb = im_a_bgr[..., ::-1]
                    pil_a = Image.fromarray(im_a_rgb)
                    
                    final_img = pil_a
                    
                    # Run B if exists
                    if model_b:
                        res_b = model_b.predict(source=media_path, save=False, conf=conf_val, iou=iou_val, stream=True)
                        im_b_bgr = next(res_b).plot()
                        im_b_rgb = im_b_bgr[..., ::-1]
                        pil_b = Image.fromarray(im_b_rgb)
                        
                        # Stitch Side-by-Side
                        w_a, h_a = pil_a.size
                        w_b, h_b = pil_b.size
                        
                        # Resize B to match A's height if needed (simple approach)
                        if h_a != h_b:
                            ratio = h_a / h_b
                            w_b = int(w_b * ratio)
                            pil_b = pil_b.resize((w_b, h_a))
                            
                        total_w = w_a + w_b
                        final_img = Image.new("RGB", (total_w, h_a))
                        final_img.paste(pil_a, (0, 0))
                        final_img.paste(pil_b, (w_a, 0))
                        
                        # Draw Labels
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(final_img)
                        # Minimal labeling
                        draw.text((10, 10), "Model A", fill="white")
                        draw.text((w_a + 10, 10), "Model B", fill="white")

                    # Save if requested
                    if save_dir:
                        fname = f"comp_{os.path.basename(media_path)}_{int(time.time())}.jpg"
                        final_img.save(os.path.join(save_dir, fname))

                    # Display in UI (Resize for simple list view)
                    base_width = 700
                    w_percent = (base_width / float(final_img.size[0]))
                    h_size = int((float(final_img.size[1]) * float(w_percent)))
                    display_img = final_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
                    
                    ctk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=(base_width, h_size))
                    
                    self.after(0, lambda img=ctk_img, name=os.path.basename(media_path): self._display_inf_result_item(img, name))
                    
            except Exception as e:
                print(e)
                self.after(0, lambda: messagebox.showerror("Batch Error", str(e)))
            finally:
                self.after(0, lambda: self.inf_run_btn.configure(state="normal", text="RUN BATCH INFERENCE"))
        
        threading.Thread(target=_run_batch, daemon=True).start()

    def _display_inf_result_item(self, ctk_img, name):
        frame = ctk.CTkFrame(self.inf_display)
        frame.pack(fill="x", padx=5, pady=5)
        
        lbl_name = ctk.CTkLabel(frame, text=name, anchor="w", font=ctk.CTkFont(weight="bold"))
        lbl_name.pack(padx=5, pady=(5,0), anchor="w")
        
        lbl_img = ctk.CTkLabel(frame, text="", image=ctk_img)
        lbl_img.pack(expand=True, padx=5, pady=5)

    def _display_inf_result(self, ctk_img):
        # Clear old
        for w in self.inf_display.winfo_children():
            w.destroy()
        
        lbl = ctk.CTkLabel(self.inf_display, text="", image=ctk_img)
        lbl.pack(expand=True, padx=10, pady=10)

    def reset_ui_state(self):
        self.after(0, lambda: self.start_button.configure(state="normal", text="START TRAINING"))

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    app = YoloTrainerApp()
    app.mainloop()
