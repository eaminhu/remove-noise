import cv2
import numpy as np
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading

# Import our text region cleaner function
from text_region_cleaner import clean_document_edges, batch_process_images

class TextCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文档图片清理工具")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.sensitivity = tk.DoubleVar(value=1.0)
        self.bg_color = tk.StringVar(value="255,255,255")
        self.processing = False
        self.current_image = None
        self.processed_image = None
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_frame, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Input folder selection
        ttk.Label(left_panel, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Entry(left_panel, textvariable=self.input_path, width=30).grid(row=1, column=0, sticky=tk.W+tk.E, pady=(0, 5))
        ttk.Button(left_panel, text="浏览...", command=self.browse_input).grid(row=1, column=1, padx=(5, 0), pady=(0, 5))
        
        # Output folder selection
        ttk.Label(left_panel, text="输出文件夹:").grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        ttk.Entry(left_panel, textvariable=self.output_path, width=30).grid(row=3, column=0, sticky=tk.W+tk.E, pady=(0, 5))
        ttk.Button(left_panel, text="浏览...", command=self.browse_output).grid(row=3, column=1, padx=(5, 0), pady=(0, 5))
        
        # Sensitivity slider
        ttk.Label(left_panel, text="文本检测灵敏度:").grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        sensitivity_slider = ttk.Scale(left_panel, from_=0.5, to=2.0, variable=self.sensitivity, orient=tk.HORIZONTAL)
        sensitivity_slider.grid(row=5, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(0, 5))
        ttk.Label(left_panel, textvariable=tk.StringVar(value="低")).grid(row=6, column=0, sticky=tk.W)
        ttk.Label(left_panel, textvariable=tk.StringVar(value="高")).grid(row=6, column=0, sticky=tk.E)
        
        # Background color
        ttk.Label(left_panel, text="背景颜色 (R,G,B):").grid(row=7, column=0, sticky=tk.W, pady=(10, 5))
        ttk.Entry(left_panel, textvariable=self.bg_color, width=15).grid(row=8, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(left_panel, text="选择颜色", command=self.choose_color).grid(row=8, column=1, padx=(5, 0), pady=(0, 5))
        
        # Process button
        process_button = ttk.Button(left_panel, text="开始处理", command=self.start_processing)
        process_button.grid(row=9, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(20, 5))
        
        # Progress bar
        self.progress = ttk.Progressbar(left_panel, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.grid(row=10, column=0, columnspan=2, sticky=tk.W+tk.E, pady=(5, 5))
        
        # Status label
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(left_panel, textvariable=self.status_var)
        status_label.grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Right panel (image preview)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image preview area
        preview_frame = ttk.LabelFrame(right_panel, text="图片预览")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(preview_frame, bg="#ffffff")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image selection
        control_frame = ttk.Frame(right_panel)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="原图/处理后", command=self.toggle_preview).pack(side=tk.LEFT, padx=5)
        
        # File list
        self.file_listbox = tk.Listbox(left_panel, height=10)
        self.file_listbox.grid(row=12, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S, pady=(10, 0))
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        
        # Add scrollbar to listbox
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=12, column=2, sticky=tk.N+tk.S, pady=(10, 0))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(12, weight=1)
    
    def browse_input(self):
        folder_path = filedialog.askdirectory(title="选择输入文件夹")
        if folder_path:
            self.input_path.set(folder_path)
            self.update_file_list()
    
    def browse_output(self):
        folder_path = filedialog.askdirectory(title="选择输出文件夹")
        if folder_path:
            self.output_path.set(folder_path)
    
    def choose_color(self):
        # Simple color chooser dialog
        color = tk.colorchooser.askcolor(title="选择背景颜色")
        if color[0]:  # color is ((r,g,b), '#rrggbb')
            r, g, b = [int(c) for c in color[0]]
            self.bg_color.set(f"{r},{g},{b}")
    
    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        input_dir = self.input_path.get()
        if not input_dir or not os.path.isdir(input_dir):
            return
        
        # Find image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
            self.image_files.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
        
        # Update listbox
        for img_file in self.image_files:
            self.file_listbox.insert(tk.END, img_file.name)
        
        # Select first image
        if self.image_files:
            self.file_listbox.selection_set(0)
            self.on_file_select(None)
    
    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if 0 <= index < len(self.image_files):
            self.load_image(self.image_files[index])
    
    def load_image(self, img_path):
        try:
            # Load original image
            self.current_image_path = img_path
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"无法读取图片: {img_path}")
            
            self.current_image = img
            self.showing_processed = False
            
            # Check if processed version exists
            output_dir = self.output_path.get()
            if output_dir:
                processed_path = Path(output_dir) / img_path.name
                if processed_path.exists():
                    self.processed_image = cv2.imread(str(processed_path))
                else:
                    self.processed_image = None
            else:
                self.processed_image = None
            
            # Display original image
            self.display_image(self.current_image)
        except Exception as e:
            messagebox.showerror("错误", f"加载图片时出错: {str(e)}")
    
    def display_image(self, img):
        if img is None:
            return
        
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:  # Canvas not yet realized
            self.root.update()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
        
        h, w = img.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # Convert to PIL format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to prevent garbage collection
    
    def toggle_preview(self):
        if self.processed_image is not None:
            if self.showing_processed:
                self.display_image(self.current_image)
                self.showing_processed = False
            else:
                self.display_image(self.processed_image)
                self.showing_processed = True
    
    def prev_image(self):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index > 0:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(index - 1)
            self.file_listbox.see(index - 1)
            self.on_file_select(None)
    
    def next_image(self):
        selection = self.file_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        if index < len(self.image_files) - 1:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(index + 1)
            self.file_listbox.see(index + 1)
            self.on_file_select(None)
    
    def start_processing(self):
        if self.processing:
            return
        
        input_dir = self.input_path.get()
        output_dir = self.output_path.get()
        
        if not input_dir:
            messagebox.showerror("错误", "请选择输入文件夹")
            return
        
        if not output_dir:
            messagebox.showerror("错误", "请选择输出文件夹")
            return
        
        # Parse background color
        try:
            bg_color = tuple(map(int, self.bg_color.get().split(',')))
            if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
                raise ValueError("背景颜色必须是三个 0-255 之间的值")
        except Exception as e:
            messagebox.showerror("错误", f"背景颜色格式错误: {str(e)}")
            return
        
        # Start processing in a separate thread
        self.processing = True
        self.status_var.set("正在处理...")
        self.progress['value'] = 0
        
        threading.Thread(target=self.process_images, args=(input_dir, output_dir, bg_color)).start()
    
    def process_images(self, input_dir, output_dir, bg_color):
        try:
            # Find all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
                image_files.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))
            
            if not image_files:
                self.root.after(0, lambda: messagebox.showinfo("信息", f"在 {input_dir} 中未找到图片文件"))
                self.root.after(0, self.reset_ui)
                return
            
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            
            # Process each image
            total = len(image_files)
            successful = 0
            
            for i, img_path in enumerate(image_files):
                try:
                    # Update progress
                    progress_value = int((i / total) * 100)
                    self.root.after(0, lambda v=progress_value: self.progress.configure(value=v))
                    self.root.after(0, lambda p=img_path.name: self.status_var.set(f"正在处理: {p}"))
                    
                    # Process image
                    output_file = Path(output_dir) / img_path.name
                    clean_document_edges(
                        str(img_path), 
                        str(output_file), 
                        bg_color=bg_color, 
                        sensitivity=self.sensitivity.get()
                    )
                    successful += 1
                    
                    # If this is the currently displayed image, reload the processed version
                    if hasattr(self, 'current_image_path') and self.current_image_path == img_path:
                        self.root.after(0, lambda: self.load_image(img_path))
                        
                except Exception as e:
                    print(f"处理 {img_path} 时出错: {str(e)}")
            
            # Update UI
            self.root.after(0, lambda: self.progress.configure(value=100))
            self.root.after(0, lambda s=successful, t=total: self.status_var.set(f"完成! 成功处理 {s}/{t} 个图片"))
            self.root.after(0, lambda s=successful, t=total: messagebox.showinfo("处理完成", f"成功处理 {s}/{t} 个图片"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理图片时出错: {str(e)}"))
        finally:
            self.root.after(0, self.reset_ui)
    
    def reset_ui(self):
        self.processing = False
        self.status_var.set("就绪")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextCleanerApp(root)
    root.mainloop()
