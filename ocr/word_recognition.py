import pytesseract
import cv2

def ocr_image(image_path):
    img = cv2.imread(image_path)
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(gray)
    return text


image_path = 'data/1.png'
extracted_text = ocr_image(image_path)
print("Extracted Text:\n", extracted_text)