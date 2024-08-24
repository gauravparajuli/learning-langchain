import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f'number of document chunks: {len(docs)}')
print(f'sample chunk:\n{docs[0].page_content}')

# function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f'\n--- creating vector store {store_name} ---')
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")
        
# 1 use open ai embeddings
# useful for general purpose embeddings with high accuracy.
print('\n--- using openAI embeddings ---')
openai_embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
create_vector_store(docs, openai_embeddings, 'chroma_db_openai')

# hugging face transformers
# ideal for leveraging a wide variety of models for different tasks
print('\n--- using hugging face transformers ---')
huggingface_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
create_vector_store(docs, huggingface_embeddings, 'chroma_db_hf')

print('embedding demonstrations for OpenAI and hugging face completed')

# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

# define the question
query = "who is odysseus' wife?"

# query each vector store
query_vector_store('chroma_db_openai', query, openai_embeddings)
query_vector_store('chroma_db_hf', query, huggingface_embeddings)

print('querying demonstrations completed')