import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Function to query a vector store
def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
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

# Define the user's question
query = "How did Juliet die?"

# showcase different retrival models
# 1. similarity search
# this method finds most similiar documents based on vector similarity
# use this when you want to retrieve top k most similiar documents
print('\n--- using similarity search ---')
query_vector_store('chroma_db_with_metadata', query, embeddings, 'similarity', {'k': 3})

# 2. max marginal relevance (MMR)
# this method balances between selecting documents that are relevant to the query and diverse among themselves
# 'fetch_k' specifies the number of documents to initially fetch based on similarity
# 'lambda_mult' controls the diversity of the results: 1 for minimu diversity, 0 for maximum
# use this when you want to avoid redundancy and retrieve diverse yet relevant documents
# Note: relevance measures how closely documents match the query
# Note: diversity ensures that retrieved documents are not too similar to each other, providing a broader range of information
print('\n--- Max Marginal Relevance ---')
query_vector_store(
    'chroma_db_with_metadata',
    query,
    embeddings,
    'mmr',
    {'k':3, 'fetch_k': 20, 'lambda_mult': 0.5}
)

# 3. similarity score threshold
# this method retrieves documents that exceed a certain similarity score threshold
# 'score_threshold' sets the minimum similarity score a document must have to be considered relevant
# use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones
print('\n--- using simlilarity score threshold ---')
query_vector_store(
    'chroma_db_with_metadata',
    query,
    embeddings,
    'similarity_score_threshold',
    {'k':3, 'score_threshold': 0.1}
)