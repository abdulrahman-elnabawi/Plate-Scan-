from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets
import io
import easyocr

# If on Windows locally:
# Load YOLOv8 model trained on license plates
model = YOLO("license_plate_detector.pt")  # Upload this model file manually in Colab
# Preprocessing function to improve OCR

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
# Run detection and OCR
reader = easyocr.Reader(['ar' , 'en'])

def detect_license_text(img_bytes):
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = np.array(pil_image)

    # Assuming `model` is a license plate detection model, this part remains unchanged.
    results = model(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    plate_texts = []

    for box in boxes:
        x1, y1, x2, y2 = box
        plate_crop = image[y1:y2, x1:x2]

        # Use EasyOCR to recognize text in the cropped plate image
        result = reader.readtext(plate_crop)

        # Extract the text from the OCR result
        text = " ".join([item[1] for item in result])
        plate_texts.append(text)

        # Optionally, draw a rectangle around the detected license plate (optional)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, plate_texts

# Upload and run

uploader = widgets.FileUpload(accept='image/*', multiple=False)
output = widgets.Output()

def on_upload(change):
    output.clear_output()
    with output:
        for _, file_info in uploader.value.items():
            img_bytes = file_info['content']
            result_img, texts = detect_license_text(img_bytes)
            display(Image.fromarray(result_img))

            if texts:
                print("Detected Plate Text(s):")
                for txt in texts:
                    print(f"→ {txt}")
            else:
                print("No license plate detected or OCR failed.")

uploader.observe(on_upload, names='value')
display(widgets.VBox([widgets.Label("Upload a car image:"), uploader, output]))

