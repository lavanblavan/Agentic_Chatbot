import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np

nltk.download('punkt')

class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_sentences_per_chunk: int = 10):
        self.model = SentenceTransformer(model_name)
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def preprocess_text(self, text):
        """
        Clean and split text into well-formed sentences.
        """
        # Remove extra spaces, weird chars
        text = re.sub(r'\s+', ' ', text).strip()
        # Sentence tokenization
        sentences = sent_tokenize(text)
        # Clean sentences
        cleaned_sentences = []
        seen = set()
        for s in sentences:
            s = s.strip()
            s = re.sub(r'\s+', ' ', s)
            if len(s) > 5 and s not in seen:
                cleaned_sentences.append(s)
                seen.add(s)
        return cleaned_sentences

    def cluster_sentences(self, sentences, min_cluster_size=2):
        """
        Cluster sentences into semantic chunks automatically.
        Uses HDBSCAN to discover the number of clusters.
        """
        if not sentences:
            return [], []

        # Compute embeddings
        embeddings = self.model.encode(sentences, convert_to_tensor=False, normalize_embeddings=True)

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)

        # Group sentences by cluster label
        clusters = {}
        for sentence, label in zip(sentences, labels):
            if label == -1:
                # Treat outliers as their own clusters
                label = f"outlier_{sentence[:15]}"
            clusters.setdefault(label, []).append(sentence)

        # Sort clusters by first occurrence in original text
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: min(sentences.index(s) for s in x[1])
        )

        # Create coarse chunks (full cluster joined)
        coarse_chunks = [' '.join(group) for _, group in sorted_clusters]
        return coarse_chunks, sorted_clusters

    def create_chunks(self, sorted_clusters):
        """
        Break clusters into sub-chunks of max_sentences_per_chunk sentences each.
        """
        all_chunks = []
        for _, group in sorted_clusters:
            for i in range(0, len(group), self.max_sentences_per_chunk):
                sub_chunk = group[i:i + self.max_sentences_per_chunk]
                chunk_text = ' '.join(sub_chunk)
                if len(chunk_text) > 30:  # skip too short noise
                    all_chunks.append(chunk_text)
        return all_chunks
    def create_chunks_group(self, sorted_clusters):
        """
        Break clusters into sub-chunks of max_sentences_per_chunk sentences each.
        """
        all_chunks = []
        all_chunk_groups = []
        for _, group in sorted_clusters:
            for i in range(0, len(group), self.max_sentences_per_chunk):
                sub_chunk = group[i:i + self.max_sentences_per_chunk]
                chunk_text = ' '.join(sub_chunk)
                  # skip too short noise
                all_chunks.append(chunk_text)
            all_chunk_groups.append(all_chunks)
            all_chunks = []

        return all_chunks

    def run(self, text):
        sentences = self.preprocess_text(text)
        coarse_chunks, sorted_clusters = self.cluster_sentences(sentences, min_cluster_size=2)

        print(f"🧠 Total sentences after preprocessing: {len(sentences)}")
        print(f"📚 Number of coarse clusters: {len(coarse_chunks)}")

        fine_chunks = self.create_chunks(sorted_clusters)
        all_chunk_groups = self.create_chunks_group(sorted_clusters)

        # print overview
        for i, chunk in enumerate(fine_chunks):
            print(f"\nChunk {i+1}: {chunk[:120]}{'...' if len(chunk) > 120 else ''}")

        return sentences, coarse_chunks, fine_chunks, all_chunk_groups


if __name__ == "__main__":
    file_path = r"C:\Users\Lavan\Desktop\Chatbot\Document_Summarizer\Document_Summarizer\Documents\Police\LK_Police_Ordinance_summary.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunker = SemanticChunker()
    sentences, coarse_chunks, fine_chunks, all_chunk_groups = chunker.run(content)

    # Example: get embeddings for fine chunks for RAG retrieval
    model = SentenceTransformer('all-MiniLM-L6-v2')
    fine_chunk_embeddings = model.encode(fine_chunks, normalize_embeddings=True)

    print(f"\n✅ Created {len(fine_chunks)} fine-grained chunks ready for indexing.")
