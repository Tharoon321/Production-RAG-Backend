import os
import google.generativeai as genai

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

MODEL = "models/text-embedding-004"


def embed_chunks(chunks):

    embeddings = []

    for chunk in chunks:

        response = genai.embed_content(
            model=MODEL,
            content=chunk.text,
            task_type="retrieval_document"
        )

        embeddings.append(
            (
                chunk.chunk_id,
                response["embedding"]
            )
        )

    return embeddings