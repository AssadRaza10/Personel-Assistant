import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):

        # FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # store text chunks
        self.text_chunks = []


    def add(self, texts, embeddings):

        embeddings = np.array(embeddings).astype("float32")

        # ensure embeddings are 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # add embeddings to FAISS
        self.index.add(embeddings)

        # store corresponding text
        self.text_chunks.extend(texts)


    def search(self, query_embedding, k=3):

        query_embedding = np.array(query_embedding).astype("float32")

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for i in indices[0]:

            if i < len(self.text_chunks):
                results.append(self.text_chunks[i])

        return results