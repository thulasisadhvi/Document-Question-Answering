
# AI-Powered Document Question-Answering System

A RESTful API that ingests documents (PDF, TXT, MD) and answers user questions using Retrieval-Augmented Generation (RAG). 

This system uses a local embedding model (`all-MiniLM-L6-v2`) to convert text into mathematical vectors, stores them in an in-memory ChromaDB vector database, and leverages an external LLM via an OpenAI-compatible endpoint to generate context-aware answers without hallucinations.

## Prerequisites

* **Python:** 3.9 or higher
* **API Key:** A free API key from Google AI Studio (or Groq/OpenAI depending on your `.env` configuration).

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/thulasisadhvi/Document-Question-Answering
cd <your-repository-directory>

```

**2. Create and activate a virtual environment**

* **macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate

```


* **Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate

```



**3. Install dependencies**

```bash
pip install -r requirements.txt

```

*(Note: Downloading the `sentence-transformers` PyTorch libraries may take a moment depending on your internet connection).*

---

## Environment Variables

This project requires external API access for the LLM generation phase.

1. Create a `.env` file in the root directory.
2. Add your API key. If you are using the free Google Gemini compatibility endpoint, it should look like this:

```env
FREE_API_KEY=your_actual_api_key_here

```

*(Note: Never commit your `.env` file to version control. A `.env.example` is provided for reference).*

---

## Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload

```

The API will be available at `http://127.0.0.1:8000`.
You can interact with the Swagger UI documentation by navigating to **`http://127.0.0.1:8000/docs`** in your browser.

---

## API Documentation & Example Usage

You can test these endpoints via the built-in `/docs` UI, Postman, or using the `curl` commands below.

### 1. Document Ingestion (`POST /upload`)

Uploads a `.txt`, `.md`, or `.pdf` file, extracts the text, chunks it, generates vector embeddings, and stores them in ChromaDB.

**Example `curl` command:**

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_document.pdf;type=application/pdf'

```

**Success Response (201 Created):**

```json
{
  "message": "File uploaded and indexed successfully."
}

```

### 2. Query Inference (`POST /query`)

Accepts a JSON payload with a user question, embeds the question, retrieves the top 3 most relevant context chunks from the database, and streams an AI-generated answer.

**Example `curl` command:**

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What are the core requirements of this project?"
}'

```

**Success Response (200 OK):**

```json
{
  "answer": "Based on the documents, the core requirements include building a functional /upload endpoint for file ingestion, validating file formats, chunking text, generating local embeddings, storing them in a vector database, and building a /query endpoint to connect to an external LLM.",
  "sources": [
    "chunk text snippet 1...",
    "chunk text snippet 2..."
  ]
}

```

### 3. Evaluation Metrics (`GET /report`)

Returns basic system evaluation metrics (currently mocked for introductory purposes) evaluating context precision and hallucination rates.

**Example `curl` command:**

```bash
curl -X 'GET' \
  '[http://127.0.0.1:8000/report](http://127.0.0.1:8000/report)' \
  -H 'accept: application/json'

```

**Success Response (200 OK):**

```json
{
  "context_precision": 0.9,
  "faithfulness": 0.85,
  "system_status": "healthy"
}

```
