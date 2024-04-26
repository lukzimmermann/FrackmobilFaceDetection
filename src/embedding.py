import numpy as np

class Embedding():
    def __init__(self, name, embeddings) -> None:
        self.name = name
        self.embeddings = embeddings
        self.embedding = np.mean(embeddings, axis=0)

    def __repr__(self) -> str:
        return f'{self.name}: {self.embedding[:10]}'
