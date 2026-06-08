from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def embed_chunks(chunks):
    texts = [chunk.text for chunk in chunks]

    vectors = model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = []

    for chunk, vector in zip(chunks, vectors):
        embeddings.append(
            (
                chunk.chunk_id,
                vector.tolist()
            )
        )

    return embeddings