import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# Initialize the embedding model globally so it only loads once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize an in-memory ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="rag_documents")

def embed_and_store(chunks: list[str], source_filename: str):
    """Generates vectors and stores chunks in ChromaDB."""
    if not chunks:
        return
    
    # The specs request manually generating embeddings and passing them to Chroma
    embeddings = model.encode(chunks).tolist()
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    
    # Store metadata alongside the chunks
    metadatas = [{"source": source_filename} for _ in range(len(chunks))]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

def semantic_search(query: str, top_k: int = 3) -> list[str]:
    """Embeds the query and fetches the most similar chunks."""
    if collection.count() == 0:
        return []
        
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # Chroma returns a list of lists for documents, we extract the first list
    if results['documents'] and len(results['documents']) > 0:
        return results['documents'][0]
    return []