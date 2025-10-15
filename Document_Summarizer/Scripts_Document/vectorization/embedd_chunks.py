import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
import faiss

nltk.download('punkt')

# ====================== SemanticChunker ======================

class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_sentences_per_chunk: int = 10):
        self.model = SentenceTransformer(model_name)
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
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
        if not sentences:
            return [], []

        embeddings = self.model.encode(sentences, convert_to_tensor=False, normalize_embeddings=True)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)

        clusters = {}
        for sentence, label in zip(sentences, labels):
            if label == -1:
                label = f"outlier_{sentence[:15]}"
            clusters.setdefault(label, []).append(sentence)

        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: min(sentences.index(s) for s in x[1])
        )

        coarse_chunks = [' '.join(group) for _, group in sorted_clusters]
        return coarse_chunks, sorted_clusters

    def create_chunks(self, sorted_clusters):
        all_chunks = []
        for _, group in sorted_clusters:
            for i in range(0, len(group), self.max_sentences_per_chunk):
                sub_chunk = group[i:i + self.max_sentences_per_chunk]
                chunk_text = ' '.join(sub_chunk)
                if len(chunk_text) > 30:
                    all_chunks.append(chunk_text)
        return all_chunks

    def create_chunks_group(self, sorted_clusters):
        all_chunk_groups = []
        for _, group in sorted_clusters:
            group_chunks = []
            for i in range(0, len(group), self.max_sentences_per_chunk):
                sub_chunk = group[i:i + self.max_sentences_per_chunk]
                chunk_text = ' '.join(sub_chunk)
                if len(chunk_text) > 30:
                    group_chunks.append(chunk_text)
            all_chunk_groups.append(group_chunks)
        return all_chunk_groups

    def run(self, text):
        sentences = self.preprocess_text(text)
        coarse_chunks, sorted_clusters = self.cluster_sentences(sentences, min_cluster_size=2)
        fine_chunks = self.create_chunks(sorted_clusters)
        all_chunk_groups = self.create_chunks_group(sorted_clusters)

        print(f"🧠 Total sentences: {len(sentences)}")
        print(f"📚 Coarse clusters: {len(coarse_chunks)}")
        print(f"✂️ Fine chunks: {len(fine_chunks)}")

        return sentences, coarse_chunks, fine_chunks, all_chunk_groups


# ====================== EmbedChunks with both retrieval ======================

class EmbedChunks:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.fine_index = None
        self.fine_chunks = None
        self.all_chunk_groups = None
        self.group_embeddings = None

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_groups(self, chunk_groups):
        group_embeddings = []
        for group in chunk_groups:
            if len(group) == 0:
                continue
            group_vecs = self.embed(group)
            group_mean = np.mean(group_vecs, axis=0)
            group_embeddings.append(group_mean)
        if not group_embeddings:
            return np.zeros((0, self.dimension), dtype='float32')
        return np.array(group_embeddings, dtype='float32')

    def build_index(self, fine_chunks, all_chunk_groups):
        self.fine_chunks = fine_chunks
        self.all_chunk_groups = all_chunk_groups

        # Fine-chunk index
        fine_embeddings = self.embed(fine_chunks).astype('float32')
        self.fine_index = faiss.IndexFlatIP(self.dimension)
        if fine_embeddings.shape[0] > 0:
            self.fine_index.add(fine_embeddings)

        # Group embeddings
        self.group_embeddings = self.embed_groups(all_chunk_groups)

    # ---------------- Flat fine-chunk retrieval ----------------
    def flat_search(self, query, top_k=5):
        query_vec = self.embed([query]).astype('float32')
        if self.fine_index is None or self.fine_index.ntotal == 0:
            return []

        # Normalize for cosine similarity
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        # FAISS expects normalized vectors for IP = cosine
        D, I = self.fine_index.search(query_vec, top_k)
        top_chunks = [self.fine_chunks[i] for i in I[0]]
        return top_chunks

    # ---------------- Hierarchical retrieval ----------------
    def hierarchical_search(self, query, top_k_groups=5, top_k_fine=5):
        query_vec = self.embed([query]).astype('float32')
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # 1️⃣ Group search
        if self.group_embeddings.shape[0] == 0:
            return []
        group_emb_norm = self.group_embeddings / np.linalg.norm(self.group_embeddings, axis=1, keepdims=True)
        scores = (group_emb_norm @ query_vec.T).squeeze()
        top_group_indices = scores.argsort()[::-1][:top_k_groups]

        # 2️⃣ Collect fine chunks from top groups
        candidate_chunks = []
        for idx in top_group_indices:
            candidate_chunks.extend(self.all_chunk_groups[idx])

        # Embed candidate fine chunks
        candidate_embeddings = self.embed(candidate_chunks).astype('float32')
        candidate_embeddings = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        sim_scores = candidate_embeddings @ query_vec.T
        top_fine_indices = sim_scores.squeeze().argsort()[::-1][:top_k_fine]

        top_chunks = [candidate_chunks[i] for i in top_fine_indices]
        return top_chunks


# ====================== MAIN ======================

if __name__ == "__main__":
    file_path = r"C:\Users\Lavan\Desktop\Chatbot\Document_Summarizer\Document_Summarizer\Documents\Police\LK_Police_Ordinance_summary.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Chunking
    chunker = SemanticChunker()
    sentences, coarse_chunks, fine_chunks, all_chunk_groups = chunker.run(content)

    # 2. Embedding + search
    embedder = EmbedChunks()
    embedder.build_index(fine_chunks, all_chunk_groups)

    query = "power of Minister to establish police force in towns"

    import time

# ---------------- Flat fine-chunk retrieval ----------------
    start_flat = time.time()
    top_flat = embedder.flat_search(query, top_k=5)
    end_flat = time.time()

    print("\n🔹 Flat fine-chunk retrieval:")
    print(f"⏱ Time taken: {end_flat - start_flat:.4f} seconds")
    for i, c in enumerate(top_flat, 1):
        print(f"{i}. {c[:150]}{'...' if len(c) > 150 else ''}")

    # ---------------- Hierarchical retrieval ----------------
    start_hier = time.time()
    top_hier = embedder.hierarchical_search(query, top_k_groups=5, top_k_fine=5)
    end_hier = time.time()

    print("\n🔹 Hierarchical retrieval:")
    print(f"⏱ Time taken: {end_hier - start_hier:.4f} seconds")
    for i, c in enumerate(top_hier, 1):
        print(f"{i}. {c[:150]}{'...' if len(c) > 150 else ''}")
