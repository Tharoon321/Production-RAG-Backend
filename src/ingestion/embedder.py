from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return _model


def embed_chunks(chunks):
    model = get_model()

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