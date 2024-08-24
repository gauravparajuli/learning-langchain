import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# define user's question
query = 'How can I learn more about LangChain?'

# retrieve the relevant documents based on the query
retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs={'k':1}
)
relevant_docs = retriever.invoke(query)

# display the relevant results with metadata
print('\n--- Relevant Documents ---')
for i, doc in enumerate(relevant_docs, 1):
    print(f'Document {i}:\n{doc.page_content}\n')

# combine the query and the relevant document contents
combined_input = (
    'here are some documents that might help answer the question: '
    + query
    + '\n\nRelevant documents:\n'
    + '\n\n'.join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

model = ChatOpenAI(model='gpt-4o-mini')

# define the messages for the model
messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content=combined_input)
]

# invoke the model with comined input
result = model.invoke(messages)

# display the full result and read content only
print('\n--- Generated response ---')
print('content only')
print(result.content)