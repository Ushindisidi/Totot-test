import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Import helper functions
from app.utils.helper import load_pdf, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_REGION = os.environ.get("PINECONE_REGION")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Embeddings model
embeddings = download_hugging_face_embeddings()
embedding_dimension = 384

# Pinecone setup
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "Totot"

if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    print("Index created. Waiting...")
    import time

    time.sleep(60)

pinecone_vectorstore = Pinecone(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

docsearch = pinecone_vectorstore.add_texts([t.page_content for t in text_chunks])
print("Embeddings created and added to Pinecone index.")
