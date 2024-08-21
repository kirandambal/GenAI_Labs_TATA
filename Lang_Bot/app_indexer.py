

import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import openai

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME")

# Initialize Azure OpenAI
openai.api_type = "azure"
openai.api_version = OPENAI_DEPLOYMENT_VERSION
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY

if __name__ == "__main__":
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
        model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
        chunk_size=1
    )
    
    fileName = "NLP.pdf"

    # Use LangChain PDF loader
    loader = PyPDFLoader(fileName)

    # Split the document into chunks
    pages = loader.load_and_split()

    # Use LangChain to create the embeddings
    db = FAISS.from_documents(documents=pages, embedding=embeddings)

    # Save the embeddings into FAISS vector store
    db.save_local("./faiss_index")
