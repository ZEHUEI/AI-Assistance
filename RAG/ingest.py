from pathlib import Path
import re
import hashlib

import chromadb
import ollama


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "RAG" / "chroma_db"

EMBEDDING_MODEL = "embeddinggemma"
COLLECTION_NAME = "zehuei_portfolio"


# ============================================================
# CHROMA DATABASE
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# Delete old collection when rebuilding the database
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )
except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# MARKDOWN CHUNKING
# ============================================================

def split_markdown(text: str):
    """
    Split Markdown primarily using headings.
    """

    sections = re.split(
        r"(?=^#{1,6}\s)",
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # If section is reasonably small,
        # keep the entire section together.
        if len(section) <= 1500:

            chunks.append(section)

        else:

            # Large section:
            # break it further by paragraphs.

            paragraphs = section.split("\n\n")

            current_chunk = ""

            for paragraph in paragraphs:

                potential_chunk = (
                    current_chunk
                    + "\n\n"
                    + paragraph
                ).strip()

                if len(potential_chunk) > 1500:

                    if current_chunk:
                        chunks.append(
                            current_chunk.strip()
                        )

                    current_chunk = paragraph

                else:

                    current_chunk = potential_chunk

            if current_chunk:
                chunks.append(
                    current_chunk.strip()
                )

    return chunks


# ============================================================
# LOAD FILES
# ============================================================

def load_documents():

    documents = []

    files = sorted(
        KNOWLEDGE_DIR.glob("*.md")
    )

    print(
        f"Found {len(files)} Markdown files.\n"
    )

    for file_path in files:

        print(
            f"Reading {file_path.name}"
        )

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = split_markdown(text)

        for index, chunk in enumerate(chunks):

            documents.append(
                {
                    "text": chunk,
                    "source": file_path.name,
                    "section": index
                }
            )

    return documents


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(documents):

    texts = [
        document["text"]
        for document in documents
    ]

    print(
        f"\nCreating embeddings for "
        f"{len(texts)} chunks..."
    )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return response.embeddings


# ============================================================
# STORE IN CHROMA
# ============================================================

def store_documents(
    documents,
    embeddings
):

    ids = []
    texts = []
    metadata = []

    for document in documents:

        unique_string = (
            f"{document['source']}-"
            f"{document['section']}"
        )

        document_id = hashlib.sha256(
            unique_string.encode()
        ).hexdigest()

        ids.append(document_id)

        texts.append(
            document["text"]
        )

        metadata.append(
            {
                "source":
                    document["source"],

                "section":
                    document["section"]
            }
        )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata
    )

    print(
        f"Stored {len(documents)} chunks."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================")
    print(" Portfolio RAG - Ingestion")
    print("==============================\n")

    documents = load_documents()

    if not documents:

        print("No Markdown files found.")
        return

    embeddings = create_embeddings(
        documents
    )

    store_documents(
        documents,
        embeddings
    )

    print("\nKnowledge database created!")
    print(
        f"Database: {CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()