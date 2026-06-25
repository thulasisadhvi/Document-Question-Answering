from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import openai

# Import our custom modules
from document_processor import extract_text, chunk_text
from vector_store import embed_and_store, semantic_search, collection
from llm_service import generate_answer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="AI-Powered Document Q&A API")
from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    # Automatically redirect users to the docs page
    return RedirectResponse(url="/docs")

# Schemas
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate extension
    allowed_extensions = ('.txt', '.md', '.pdf')
    if not file.filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload .txt, .md, or .pdf"
        )
    
    # 2. Read file
    file_bytes = await file.read()
    
    # 3. Extract text
    text = extract_text(file_bytes, file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="Document contains no readable text.")
        
    # 4. Chunk text
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    
    # 5. Embed and Store
    try:
        embed_and_store(chunks, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process and store embeddings: {str(e)}")
        
    return {"message": "File uploaded and indexed successfully."}

@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents have been indexed yet.")

    # 1. Semantic Search
    relevant_chunks = semantic_search(question, top_k=3)
    
    # 2. LLM Generation
    try:
        answer = generate_answer(question, relevant_chunks)
    except openai.APIConnectionError:
        raise HTTPException(status_code=502, detail="Bad Gateway: External LLM API is unreachable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
    # 3. Return formatted response
    return QueryResponse(
        answer=answer,
        sources=relevant_chunks
    )

@app.get("/report", status_code=status.HTTP_200_OK)
async def get_report():
    """Returns hardcoded evaluation metrics as requested."""
    return {
        "context_precision": 0.90,
        "faithfulness": 0.85,
        "system_status": "healthy"
    }