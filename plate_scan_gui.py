import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import io

class ModernStyle:
    # Color scheme
    PRIMARY_COLOR = "#011526"  # Dark navy blue
    SECONDARY_COLOR = "#1e3d59"  # Medium navy blue
    BG_COLOR = "#011526"  # Dark navy blue for main background
    FRAME_BG_COLOR = "#0a2942"  # Slightly lighter navy blue for frames
    TEXT_BG_COLOR = "#FFFFFF"  # Navy blue for text background
    TEXT_COLOR = "#000000"  # White text for better contrast
    ACCENT_COLOR = "#4a90e2"  # Bright blue for highlights
    SUCCESS_COLOR = "#2ecc71"  # Green for success messages
    
    # Fonts
    TITLE_FONT = ("Helvetica", 16, "bold")
    BUTTON_FONT = ("Helvetica", 12, "bold")
    TEXT_FONT = ("Helvetica", 10)
    
    # Dimensions
    BUTTON_PADDING = (20, 12)
    FRAME_PADDING = 15

class LicensePlateDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plate Scan Pro")
        self.root.geometry("1000x800")
        self.style = ModernStyle()
        
        # Configure the root window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize the YOLO model and EasyOCR
        self.model = YOLO("license_plate_detector.pt")
        self.reader = easyocr.Reader(['ar', 'en'])
        
        # Setup custom styles
        self.setup_styles()
        
        # Create GUI elements
        self.create_widgets()
        
    def setup_styles(self):
        style = ttk.Style()
        
        # Configure main styles
        style.configure("Main.TFrame",
                       background=self.style.FRAME_BG_COLOR)
        
        # Configure custom button style
        style.configure("Custom.TButton",
                       padding=self.style.BUTTON_PADDING,
                       font=self.style.BUTTON_FONT,
                       background=self.style.SECONDARY_COLOR)
        
        # Configure custom label style
        style.configure("Custom.TLabel",
                       background=self.style.FRAME_BG_COLOR,
                       font=self.style.TEXT_FONT,
                       foreground=self.style.TEXT_COLOR)
        
        # Configure title label style
        style.configure("Title.TLabel",
                       font=self.style.TITLE_FONT,
                       foreground=self.style.TEXT_COLOR,
                       background=self.style.FRAME_BG_COLOR,
                       padding=(0, 10))
        
    def create_widgets(self):
        # Set window background
        self.root.configure(bg=self.style.BG_COLOR)
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=self.style.FRAME_PADDING, style="Main.TFrame")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=3)  # Give more weight to image row
        
        # Add title
        title_label = ttk.Label(
            main_frame,
            text="Plate Scan Pro",
            style="Title.TLabel"
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Create buttons frame
        button_frame = ttk.Frame(main_frame, style="Main.TFrame")
        button_frame.grid(row=1, column=0, pady=(0, 20), sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        
        # Add Browse button
        self.browse_button = ttk.Button(
            button_frame,
            text="Select Image",
            command=self.load_image,
            style="Custom.TButton"
        )
        self.browse_button.grid(row=0, column=0)
        
        # Create image container frame
        self.image_container = ttk.Frame(main_frame, style="Main.TFrame")
        self.image_container.grid(row=2, column=0, sticky="nsew")
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)
        
        # Create canvas for scrollable image
        self.image_canvas = tk.Canvas(
            self.image_container,
            bg=self.style.BG_COLOR,
            highlightthickness=0
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbars for the canvas
        y_scrollbar = ttk.Scrollbar(
            self.image_container,
            orient="vertical",
            command=self.image_canvas.yview
        )
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        
        x_scrollbar = ttk.Scrollbar(
            self.image_container,
            orient="horizontal",
            command=self.image_canvas.xview
        )
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure canvas scrolling
        self.image_canvas.configure(
            xscrollcommand=x_scrollbar.set,
            yscrollcommand=y_scrollbar.set
        )
        
        # Create frame inside canvas for the image
        self.image_frame = tk.Frame(
            self.image_canvas,
            bg=self.style.BG_COLOR
        )
        
        # Create image label with centering
        self.image_label = ttk.Label(
            self.image_frame,
            style="Custom.TLabel"
        )
        self.image_label.pack(expand=True, fill="both", padx=2, pady=2)
        
        # Add the image frame to the canvas
        self.canvas_frame = self.image_canvas.create_window(
            (0, 0),
            window=self.image_frame,
            anchor="center",
            tags="frame"
        )
        
        # Bind events for scrolling and resizing
        self.image_frame.bind("<Configure>", self.on_frame_configure)
        self.image_canvas.bind("<Configure>", self.on_canvas_configure)
        self.root.bind("<Configure>", self.on_window_resize)
        
        # Store the current image for resizing
        self.current_image = None
        
        # Create results frame with scrollbar
        results_frame = ttk.Frame(main_frame, style="Main.TFrame")
        results_frame.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Add results label
        results_label = ttk.Label(
            results_frame,
            text="Detection Results",
            style="Title.TLabel"
        )
        results_label.grid(row=0, column=0, pady=(0, 10))
        
        # Create text display with custom styling
        self.text_display = tk.Text(
            results_frame,
            height=5,
            width=50,
            font=self.style.TEXT_FONT,
            bg=self.style.TEXT_BG_COLOR,
            fg=self.style.TEXT_COLOR,
            wrap=tk.WORD,
            padx=10,
            pady=10,
            relief="flat",
            borderwidth=0,
            state="disabled",  # Make it read-only
            cursor="arrow"  # Change cursor to normal arrow instead of text cursor
        )
        self.text_display.grid(row=1, column=0, sticky="ew")
        
        # Add scrollbar to text display
        text_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.text_display.yview)
        text_scrollbar.grid(row=1, column=1, sticky="ns")
        self.text_display.configure(yscrollcommand=text_scrollbar.set)
        
        # Add status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            style="Custom.TLabel"
        )
        self.status_bar.grid(row=4, column=0, pady=(10, 0), sticky="ew")
    
    def on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def on_canvas_configure(self, event=None):
        """When canvas is resized, update the window position"""
        if event is not None:
            # Get the canvas dimensions
            canvas_width = event.width
            canvas_height = event.height
            
            # Update the position of the frame to keep it centered
            self.image_canvas.coords(
                self.canvas_frame,
                canvas_width/2,
                canvas_height/2
            )
            
            # Update the scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def on_window_resize(self, event=None):
        """Handle window resize events"""
        if self.current_image and hasattr(self, 'image_container'):
            # Get current container size
            width = self.image_container.winfo_width() - 20
            height = self.image_container.winfo_height() - 20
            
            if width > 0 and height > 0:
                # Calculate new size maintaining aspect ratio
                img_width, img_height = self.current_image.size
                ratio = min(width/img_width, height/img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # Resize the image
                resized_image = self.current_image.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
                
                # Update the display
                photo = ImageTk.PhotoImage(resized_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
                # Center the image in the canvas
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                
                # Update the canvas window position to center
                self.image_canvas.coords(
                    self.canvas_frame,
                    canvas_width/2,
                    canvas_height/2
                )
                
                # Update the canvas scroll region
                self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                self.status_var.set("Processing image...")
                self.root.update()
                
                # Read and process the image
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect license plates
                results = self.model(image)[0]
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                
                # Draw rectangles and get texts
                plate_texts = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    plate_crop = image[y1:y2, x1:x2]
                    
                    # Use EasyOCR
                    result = self.reader.readtext(plate_crop)
                    # Process the text to remove "Egypt" and "مصر"
                    processed_text = []
                    for item in result:
                        text = item[1]
                        # Remove "Egypt" and "مصر" (case insensitive)
                        text = text.replace("Egypt", " ").replace("EGYPT", " ").replace("E", " ").replace("G", " ").replace("Y", " ").replace("P", " ").replace("T", " ")               
                        text = text.replace("مصر", " ")
                        # Remove any extra spaces
                        text = " ".join(text.split())
                        if text:  # Only add non-empty text
                            processed_text.append(text)
                    
                    text = " ".join(processed_text)
                    if text.strip():  # Only add non-empty strings
                        plate_texts.append(text)
                    
                    # Draw rectangle with custom color
                    cv2.rectangle(image, (x1, y1), (x2, y2), 
                                tuple(int(x) for x in hex_to_rgb(self.style.SUCCESS_COLOR)), 2)
                
                # Convert to PIL Image and store as current_image
                self.current_image = Image.fromarray(image)
                
                # Trigger initial resize
                self.on_window_resize()
                
                # Update canvas scrollregion
                self.image_frame.update_idletasks()
                self.on_frame_configure()
                
                # Display detected texts
                self.text_display.configure(state="normal")  # Temporarily enable for updating
                self.text_display.delete(1.0, tk.END)
                if plate_texts:
                    self.text_display.insert(tk.END, "📋 Detected License Plates:\n\n")
                    for i, text in enumerate(plate_texts, 1):
                        if text.strip():  # Only display non-empty text
                            self.text_display.insert(tk.END, f"🔍 Plate {i}: {text}\n")
                    self.status_var.set(f"Successfully detected {len(plate_texts)} license plate(s)")
                else:
                    self.text_display.insert(tk.END, "❌ No license plates detected in the image.")
                    self.status_var.set("No license plates detected")
                self.text_display.configure(state="disabled")  # Make read-only again
                    
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.text_display.configure(state="normal")  # Temporarily enable for updating
                self.text_display.delete(1.0, tk.END)
                self.text_display.insert(tk.END, f"❌ Error processing image: {str(e)}")
                self.text_display.configure(state="disabled")  # Make read-only again

def hex_to_rgb(hex_color):
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def main():
    root = tk.Tk()
    app = LicensePlateDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 