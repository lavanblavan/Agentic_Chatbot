from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows

class TextExtractor:
    def __init__(self, lang='eng'):
        self.lang = lang

    def image_to_text(self, image: Image.Image) -> str:
        """
        Extract text from a PIL image using Tesseract OCR
        """
        
        text = pytesseract.image_to_string(image, lang=self.lang)
        print(f"Extracted Text: {text[:100]}...")  # Print first 100 chars
        return text

    def images_to_texts(self, images):
        """
        Extract text from a list of PIL images
        Returns list of strings (one per image)
        """
        texts = [self.image_to_text(img) for img in images]
        return texts