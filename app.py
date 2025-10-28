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
system_prompt='''
You are CyberGuard — an AI-powered Cybercrime Awareness Assistant designed to educate and assist Indian citizens in understanding cyber threats, prevention methods, and reporting procedures.

Your knowledge base is built from official resources, including:
- Cyber Security Awareness Booklet for Citizens (India)
- Types of Cyber Crimes (India)
- Cyber Security Tips by Cyber Crime Authorities
- Types of Cyber Crimes and Prevention Measures

Your responsibilities:
1. Provide clear, simple, and accurate answers about cybercrime awareness, prevention, and safety.
2. Reference relevant sections from the provided documents (metadata) when possible.
3. Use an informative yet friendly tone — suitable for all citizens.
4. If a query is outside the scope of cyber safety or Indian cyber law, politely decline and redirect users to appropriate authorities or helplines.


Never generate or infer sensitive or false information.
Your goal is to educate and empower citizens to stay safe online.
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
])

# Retrieve relevant text from vector DB
 retrieved_docs = vector_db.similarity_search("{query}",k=3)

# ✅ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",retriever=retrieved_docs, temperature=0.2)
parser = StrOutputParser()



chain=prompt|llm|parser

# ✅ Create FastAPI app
app = FastAPI(title="CyberCrime Chatbot API")


class Query(BaseModel):
    query: str = Field(..., example="What are the common types of cyber crimes in India?")


@app.post("/chat")
def chat_endpoint(query: Query):
    # Step 1: Retrieve relevant text from vector DB
    resulted_text=chain.invoke({"query": query.query})
    return {"user_query": query.query, "bot_response": resulted_text}


@app.get("/")
def root():
    return {"message": "✅ Cyber Crime Chatbot API is running successfully!"}


# ✅ Uncomment to run locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
