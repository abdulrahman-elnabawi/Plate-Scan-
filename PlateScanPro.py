from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageTk
import io
import easyocr
import os
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

# Load YOLOv8 model trained on license plates
model = YOLO("C:/Users/interface/Desktop/New folder (8)/license_plate_detector.pt")

def preprocess_and_ocr(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)
    plate_filtered = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    plate_thresh = cv2.adaptiveThreshold(
        plate_filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    plate_resized = cv2.resize(plate_thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(plate_resized, config='--psm 7')
    return text.strip()

# Initialize EasyOCR reader with both English and Arabic support
reader = easyocr.Reader(['ar', 'en'])

def clean_text(text):
    # Remove "egypt" and "مصر" (case insensitive)
    text = text.lower()
    text = text.replace("egypt", " ").replace("مصر", " ").replace("gypt", " ").replace("E", " ").replace("g", " ").replace("y", " ").replace("p", " ").replace("t", " ")
    # Remove extra spaces
    text = " ".join(text.split())
    return text.strip()

def detect_license_text(img_bytes):
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = np.array(pil_image)

    results = model(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    plate_texts = []

    for box in boxes:
        x1, y1, x2, y2 = box
        plate_crop = image[y1:y2, x1:x2]

        # Use EasyOCR to recognize text in the cropped plate image
        result = reader.readtext(plate_crop)

        # Extract the text from the OCR result and clean it
        text = " ".join([item[1] for item in result])
        cleaned_text = clean_text(text)
        
        if cleaned_text.strip():  # Only add non-empty text
            plate_texts.append(cleaned_text)
            # Draw a rectangle around the detected license plate
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, plate_texts

class LicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")
        
        # Set the background color
        self.root.configure(bg='#2C2F33')  # Dark gray background
        
        # Configure styles
        self.setup_styles()
        
        # Configure root grid
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20", style='Main.TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=20, pady=20)
        
        # Configure main frame grid
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=3)  # Image gets more space
        self.main_frame.grid_rowconfigure(2, weight=1)  # Text gets less space
        
        # Create widgets
        self.create_widgets(self.main_frame)

    def setup_styles(self):
        # Create custom styles
        style = ttk.Style()
        
        # Configure main frame style
        style.configure('Main.TFrame', background='#2C2F33')
        
        # Configure button style
        style.configure('Custom.TButton',
                       padding=(20, 10),
                       font=('Helvetica', 12, 'bold'))
        
        # Configure label frame style
        style.configure('Custom.TLabelframe',
                       background='#23272A',  # Darker gray for contrast
                       foreground='white')
        style.configure('Custom.TLabelframe.Label',
                       background='#23272A',
                       foreground='white',
                       font=('Helvetica', 11, 'bold'))
        
        # Configure canvas style (will be used as background)
        style.configure('Custom.Canvas',
                       background='#23272A')
        
    def create_widgets(self, frame):
        # Create centered button frame
        button_frame = ttk.Frame(frame, style='Main.TFrame')
        button_frame.grid(row=0, column=0, pady=20, sticky=(tk.N, tk.E, tk.W))
        button_frame.grid_columnconfigure(0, weight=1)  # Center the button
        
        # Upload button with custom style
        self.upload_btn = ttk.Button(
            button_frame,
            text="Upload Image",
            command=self.upload_image,
            style='Custom.TButton'
        )
        self.upload_btn.grid(row=0, column=0)
        
        # Image display frame with custom style
        self.image_frame = ttk.LabelFrame(
            frame,
            text="Image",
            padding="10",
            style='Custom.TLabelframe'
        )
        self.image_frame.grid(row=1, column=0, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Configure image frame grid
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)
        
        # Create canvas with custom background
        self.image_canvas = tk.Canvas(
            self.image_frame,
            bg='#23272A',  # Dark background for contrast
            highlightthickness=0  # Remove border
        )
        self.image_scrollbar_y = ttk.Scrollbar(
            self.image_frame,
            orient="vertical",
            command=self.image_canvas.yview
        )
        self.image_scrollbar_x = ttk.Scrollbar(
            self.image_frame,
            orient="horizontal",
            command=self.image_canvas.xview
        )
        
        # Configure image canvas scrolling
        self.image_canvas.configure(
            xscrollcommand=self.image_scrollbar_x.set,
            yscrollcommand=self.image_scrollbar_y.set
        )
        
        # Grid image canvas and scrollbars
        self.image_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.image_scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.image_scrollbar_x.grid(row=1, column=0, sticky=(tk.E, tk.W))
        
        # Results display frame
        self.result_frame = ttk.LabelFrame(
            frame,
            text="Detected License Plates",
            padding="10",
            style='Custom.TLabelframe'
        )
        self.result_frame.grid(row=2, column=0, pady=10, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Configure result frame grid
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=1)
        
        # Results text widget with custom colors
        self.result_text = ScrolledText(
            self.result_frame,
            wrap=tk.WORD,
            height=8,
            font=('Helvetica', 10),
            bg='#2F3136',  # Dark gray background
            fg='white',    # White text
            insertbackground='white'  # White cursor
        )
        self.result_text.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
        )
        
        if file_path:
            # Read the image
            with open(file_path, 'rb') as f:
                img_bytes = f.read()
            
            # Process the image
            result_img, texts = detect_license_text(img_bytes)
            
            # Display the image
            img = Image.fromarray(result_img)
            
            # Calculate scaling factor to fit in canvas while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            img_width, img_height = img.size
            
            # Calculate scaling factors
            width_factor = canvas_width / img_width
            height_factor = canvas_height / img_height
            scale_factor = min(width_factor, height_factor, 1.0)  # Don't enlarge images
            
            # Resize image if needed
            if scale_factor < 1.0:
                new_width = int(img_width * scale_factor)
                new_height = int(img_height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.image_canvas.delete("all")
            
            # Calculate position to center the image
            x = max(0, (canvas_width - img.width) // 2)
            y = max(0, (canvas_height - img.height) // 2)
            
            # Create image on canvas
            self.image_canvas.create_image(x, y, anchor="nw", image=photo)
            self.image_canvas.image = photo
            
            # Update canvas scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
            # Display results with custom formatting
            self.result_text.delete(1.0, tk.END)
            if texts:
                self.result_text.insert(tk.END, "📋 Detected License Plates:\n\n", "header")
                for i, text in enumerate(texts, 1):
                    self.result_text.insert(tk.END, f"🔍 {i}. ", "bullet")
                    self.result_text.insert(tk.END, f"{text}\n\n", "text")
            else:
                self.result_text.insert(tk.END, "❌ No license plates detected or OCR failed.", "error")
            
            # Configure text tags for styling
            self.result_text.tag_configure("header", font=('Helvetica', 11, 'bold'))
            self.result_text.tag_configure("bullet", font=('Helvetica', 10, 'bold'))
            self.result_text.tag_configure("text", font=('Helvetica', 10))
            self.result_text.tag_configure("error", font=('Helvetica', 10, 'italic'))

def main():
    root = tk.Tk()
    app = LicensePlateGUI(root)
    # Set initial window size
    root.geometry("1000x800")
    # Set minimum window size
    root.minsize(600, 400)
    root.mainloop()

if __name__ == "__main__":
    main()


