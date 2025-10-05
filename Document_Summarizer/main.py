import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Settings.Settings import Config
from Scripts_Document.Extraction import TextExtractor
from Scripts_Document.preprocess import DocumentPreprocessor


class DocumentSummarizer:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.preprocessor = DocumentPreprocessor()
        self.documents = []
        self.config = Config()

    
    def get_pdf(self, pdf_path):
        """
        Load Documents
        """
        pdf_folder_path = Config.documents_folder 
        print("PDF Folder Path:", pdf_folder_path)
        for folder in os.listdir(pdf_folder_path):
            print("Folder in documents folder:", folder)
            for file in os.listdir(os.path.join(pdf_folder_path, folder)):
                
                if file.lower().endswith(".pdf"):
                    full_path = os.path.join(pdf_folder_path,folder, file)
                    self.documents.append(full_path)
        print(f"Found {len(self.documents)} PDF documents.")
        return self.documents
    def main(self):
        pdfs = self.get_pdf(Config.documents_folder)
        for pdf in pdfs:
            print(f"Processing {pdf}...")
            preprocessed_images = self.preprocessor.process_pdf(pdf)
            text = self.text_extractor.images_to_texts(preprocessed_images)
            print(f"Extracted Text from {pdf}:\n{text[:500]}...")  # Print first 500 chars

if __name__ == "__main__":
    summarizer = DocumentSummarizer()
    summarizer.main()