import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
from pathlib import Path
import json
import shutil
from PIL import Image, ImageTk
from main import IDCardProcessor

class IDCardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Egyptian ID Card Processor")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')  # Light gray background
        
        # Apply a theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), padding=10)
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Result.TLabel', font=('Helvetica', 11))
        style.configure('Action.TButton', font=('Helvetica', 11), padding=5)
        
        self.selected_image_path = None
        self.setup_gui()
        self.processor = IDCardProcessor()

    def setup_gui(self):
        # Main container
        self.main_container = ttk.Frame(self.root, padding="20")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(
            self.main_container, 
            text="Egyptian ID Card Information Extractor",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Left panel for upload and preview
        left_panel = ttk.Frame(self.main_container)
        left_panel.grid(row=1, column=0, sticky="nw", padx=10)

        # Upload section
        upload_frame = ttk.LabelFrame(
            left_panel, 
            text="Upload ID Card",
            padding="10"
        )
        upload_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))

        # Upload button and file name
        self.upload_btn = ttk.Button(
            upload_frame,
            text="Select Image",
            command=self.upload_image,
            style='Action.TButton'
        )
        self.upload_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.file_label = ttk.Label(
            upload_frame,
            text="No file selected",
            style='Result.TLabel'
        )
        self.file_label.grid(row=0, column=1, padx=5, pady=5)

        # Preview section
        preview_frame = ttk.LabelFrame(
            left_panel,
            text="Preview",
            padding="10"
        )
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 20))

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, padx=10, pady=10)

        # Process button
        self.process_btn = ttk.Button(
            left_panel,
            text="Process ID Card",
            command=self.process_image,
            style='Action.TButton'
        )
        self.process_btn.grid(row=2, column=0, pady=10)
        self.process_btn.state(['disabled'])

        # Right panel for results
        right_panel = ttk.Frame(self.main_container)
        right_panel.grid(row=1, column=1, sticky="ne", padx=10)

        # Results section
        self.results_frame = ttk.LabelFrame(
            right_panel,
            text="Extracted Information",
            padding="15"
        )
        self.results_frame.grid(row=0, column=0, sticky="nsew")

        # Status message
        self.status_label = ttk.Label(
            self.main_container,
            text="Ready",
            style='Result.TLabel'
        )
        self.status_label.grid(row=2, column=0, columnspan=2, pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select ID Card Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.file_label.config(text=Path(file_path).name)
            self.process_btn.state(['!disabled'])  # Enable process button
            self.show_preview(file_path)

    def show_preview(self, image_path):
        try:
            image = Image.open(image_path)
            # Calculate resize ratio to fit in 500x400 box
            ratio = min(500/image.width, 400/image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preview: {str(e)}")

    def show_results(self, results):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Headers
        headers = {
            'first_name': 'First Name',
            'last_name': 'Last Name',
            'address': 'Address',
            'id_number': 'ID Number'
        }

        # Display results in a more organized way
        row = 0
        for key, value in results.items():
            # Header
            header = ttk.Label(
                self.results_frame,
                text=headers.get(key, key.replace('_', ' ').title()) + ":",
                style='Header.TLabel'
            )
            header.grid(row=row, column=0, sticky="w", padx=5, pady=(10, 2))
            
            # Value
            value_label = ttk.Label(
                self.results_frame,
                text=value,
                style='Result.TLabel',
                wraplength=300  # Wrap long text
            )
            value_label.grid(row=row+1, column=0, sticky="w", padx=20, pady=(0, 10))
            
            row += 2

        # Add button to open output folder
        ttk.Button(
            self.results_frame,
            text="Open Output Folder",
            command=lambda: os.startfile(self.processor.output_folder),
            style='Action.TButton'
        ).grid(row=row, column=0, pady=20)

    def update_status(self, message, is_error=False):
        self.status_label.config(
            text=message,
            foreground='red' if is_error else 'green'
        )

    def process_image(self):
        if not self.selected_image_path:
            messagebox.showerror("Error", "Please select an image first!")
            return
            
        self.update_status("Processing...")
        self.process_btn.state(['disabled'])
        
        try:
            test_image_path = Path('test.jpg')
            shutil.copy2(self.selected_image_path, test_image_path)
            
            results = self.processor.process_image(str(test_image_path))
            self.processor.cleanup()
            
            self.show_results(results)
            self.update_status("Processing completed successfully!")
            
        except ValueError as ve:
            self.update_status(f"Error: {str(ve)}", is_error=True)
            messagebox.showerror("Error", f"Processing failed: {str(ve)}")
        except Exception as e:
            self.update_status(f"Error: {str(e)}", is_error=True)
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
        finally:
            self.process_btn.state(['!disabled'])
            
            if test_image_path.exists():
                test_image_path.unlink()

def main():
    root = tk.Tk()
    app = IDCardGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 