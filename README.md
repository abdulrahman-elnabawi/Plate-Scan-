# 🚘 Plate Scan  – License Plate Detection and OCR System

**Plate Scan Pro** is an intelligent application that detects license plates from images and extracts their text using deep learning (YOLOv8) and Optical Character Recognition (OCR) with support for both Arabic and English languages. The project also features an interactive dark-themed GUI built using Tkinter.

---

## 📌 Features

- 🔍 **YOLOv8 License Plate Detection**
  - Detects license plates in any uploaded image with high accuracy.
  
- 🧠 **OCR with EasyOCR & Tesseract**
  - Extracts and cleans text from detected plates.
  - Supports Arabic and English characters.

- 🖼️ **Image Preprocessing**
  - Enhances the plate image using grayscale conversion, filtering, and adaptive thresholding before OCR.

- 💻 **User Interface (GUI)**
  - Upload and display images through an intuitive GUI.
  - Interactive canvas with scrollbars and dynamic image resizing.
  - Stylish dark mode design for better UX.

- 📋 **Text Output**
  - Displays detected license plate text in a formatted ScrolledText widget.
  - Automatically filters out words like "Egypt" or "مصر" for cleaner results.

---

## 🛠️ Technologies Used

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [Python Pillow (PIL)](https://pillow.readthedocs.io/)

---

## 🖥️ How to Run

### 1. Clone the Repository

```bash
git clone 
## 🖥️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/platescanpro.git
cd platescanpro


