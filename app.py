# -*- coding: utf-8 -*-
"""Cyber Crime Chatbot API"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
import os

# LangChain + Google GenAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ✅ Load API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GOOGLE_API_KEY or not HF_API_KEY:
    print("⚠️ Warning: Missing API keys. Set GOOGLE_API_KEY and HUGGINGFACEHUB_API_TOKEN.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY


# ✅ Load Chroma Vector DB
if not os.path.exists("cyber_crimedb1"):
    os.makedirs("cyber_crimedb1")

PERSIST_DIR = "cyber_crimedb1"

vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name="cyber_crime1"
)


# ✅ System Prompt
system_prompt = """
You are CyberGuard — an AI-powered Cybercrime Awareness Assistant designed to educate and assist Indian citizens in understanding cyber threats, prevention methods, and reporting procedures.

Your knowledge base includes:
- Cyber Security Awareness Booklet for Citizens (India)
- Types of Cyber Crimes (India)
- Cyber Security Tips by Indian Cybercrime Authorities

Responsibilities:
1. Provide clear, accurate answers about cybercrime awareness, prevention, and safety.
2. Reference the relevant document when possible.
3. Maintain a friendly, informative tone.
4. If asked outside your scope, politely decline and guide users to official helplines.

Never provide false or unsafe information.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{context}\n\nUser Query: {query}")
])

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
parser = StrOutputParser()


# ✅ Create FastAPI app
app = FastAPI(title="CyberCrime Chatbot API")


class Query(BaseModel):
    query: str = Field(..., example="What are the common types of cyber crimes in India?")


@app.post("/chat")
def chat_endpoint(request: Query):
    # Step 1: Retrieve relevant text from vector DB
    retrieved_docs = vector_db.similarity_search(request.query, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Step 2: Generate response
    final_prompt = prompt.format(context=context, query=request.query)
    response = llm.invoke(final_prompt)
    final_output = parser.parse(response.content)

    return {"user_query": request.query, "bot_response": final_output}


@app.get("/")
def root():
    return {"message": "✅ Cyber Crime Chatbot API is running successfully!"}


# ✅ Uncomment to run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
