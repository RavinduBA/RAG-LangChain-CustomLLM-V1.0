from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Same directory and embedding model as in createdatabase.py
CHROMA_PATH = "chroma"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load the embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

# Load the Chroma DB with the embedding function
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Perform similarity search
query = "How does Alice meet the Mad Hatter?"
docs = db.similarity_search(query, k=5)

# Print results
if docs:
    for i, d in enumerate(docs):
        print(f"\n--- Result {i+1} ---\n{d.page_content}")
else:
    print("No matching documents found.")
