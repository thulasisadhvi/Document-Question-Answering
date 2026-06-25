import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Point the OpenAI client to Google's free API endpoint
client = OpenAI(
    api_key=os.getenv("FREE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def generate_answer(question: str, context_chunks: list[str]) -> str:
    context_string = "\n---\n".join(context_chunks)
    
    prompt = f"""You are a helpful assistant answering questions based on the provided document context.

Context Information:
---------------------
{context_string}
---------------------

Given the context information and no prior knowledge, answer the following user question. 
If the answer is not contained in the context, explicitly state "I cannot find the answer in the provided documents."

Question: {question}
Answer:"""

    # Upgraded from the deprecated 1.5 model to the active 2.5 model
    response = client.chat.completions.create(
        model="gemini-2.5-flash", 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()