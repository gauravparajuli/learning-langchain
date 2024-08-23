import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# define the embedding model
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# define the user's question
query = "Who is Odysseus' wife?"

# retrieve relevant docuents based on the query
retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs=dict(k=3, score_threshold=0.4)
)

relevant_docs = retriever.invoke(query)

# display the relevant result with metadata
print('--- relevant documents ---')
for i, doc in enumerate(relevant_docs, 1):
    print(f'document {i}:\n{doc.page_content}\n')
    if doc.metadata:
        print(f"source: {doc.metadata.get('source', 'Unknown')}\n")