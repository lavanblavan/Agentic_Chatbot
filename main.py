import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Settings.Settings import Config
from Scripts_Document.Extraction import TextExtractor
from Scripts_Document.preprocess import DocumentPreprocessor
from Scripts_Document.Summary_creator import summary_create 


class DocumentSummarizer:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.preprocessor = DocumentPreprocessor()
        self.summary_creator = summary_create()
        self.documents = []
        self.config = Config()

    def get_txt(self):
        """
        Load Documents
        """
        txt_folder_path = Config.documents_folder 
        print("TXT Folder Path:", txt_folder_path)
        for folder in os.listdir(txt_folder_path):
            print("Folder in documents folder:", folder)
            for file in os.listdir(os.path.join(txt_folder_path, folder)):
                
                if file.lower().endswith(".txt"):
                    full_path = os.path.join(txt_folder_path,folder, file)
                    self.documents.append(full_path)
        print(f"Found {len(self.documents)} TXT documents.")
        return self.documents

    
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
            array_of_summaries = []
            print(f"Processing {pdf}...")
            preprocessed_images = self.preprocessor.process_pdf(pdf)
            text = self.text_extractor.images_to_texts(preprocessed_images)
            for i in text:
                summary = self.summary_creator.find_minititles(i)
                array_of_summaries.append(summary)
                print(f"Summary for a page:\n{summary}\n")

            print(f"Extracted {len(text)} pages of text from {pdf}.")
            final_summary = "\n".join(array_of_summaries)
            document_name = Path(pdf).parent / (Path(pdf).stem + "_summary.txt")
            self.summary_creator.put_summary(document_name, final_summary)

        txts = self.get_txt()

# Group text files by their parent folder
        folders = {}
        for txt in txts:
            folder = Path(txt).parent
            folders.setdefault(folder, []).append(txt)

        # Process each folder and create one combined summary file
        for folder, files in folders.items():
            print(f"\nProcessing folder: {folder.name}")
            folder_summaries = []

            for txt in files:
                print(f"Processing {txt}...")
                preprocessed_text = self.preprocessor.process_txt(txt)
                text = self.text_extractor.images_to_texts(preprocessed_text)

                for i in text:
                    summary = self.summary_creator.find_minititles(i)
                    folder_summaries.append(summary)
                    print(f"Summary for a section in {txt}:\n{summary}\n")

                print(f"Extracted {len(text)} sections of text from {txt}.")

            # Combine summaries for this folder
            final_summary = "\n".join(folder_summaries)
            summary_filename = f"{folder.name}_summary.txt"
            summary_path = folder / summary_filename

            # Save the combined summary in the same folder
            self.summary_creator.put_summary(summary_path, final_summary)
            print(f"✅ Folder summary saved: {summary_path}")

if __name__ == "__main__":
    summarizer = DocumentSummarizer()
    summarizer.main()