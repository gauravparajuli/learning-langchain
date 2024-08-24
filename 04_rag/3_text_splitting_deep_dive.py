import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# read the text content from file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Define the embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # Update to a valid embedding model if needed

# function to create and persist vector store
def create_vector_store(docs, store_name):
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
        
# 1. character based splitting
# splits text into chunks based on a specified number of characters
# useful for consistent chunk sizes regardless of content structure
print(f'\n--- character based splitting ---')
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, 'chroma_db_char')

# 2. sentence based splitting
# splits text into chunks based on sentences, ensuring chunks end at sentence boundaries
# ideal for maintaining semantic coherence within chunks
print(f'\n--- sentence based splitting ---')
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, 'chroma_db_sent')

# 3 token based splitting
# splits texts into chunks based on tokens (words or subwords) using tokenizer like gpt2
# useful for transformer models with strict token limits
print(f'\n--- token based splitting ---')
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, 'chroma_db_token')

# 4 recursive character based splitting
# attempts to split text at natural boundaries (sentences, paragraphs) within character limit
# balances between maintaining coherence and adhering to character limits.
print(f'\n--- recursive based character splitting ---')
rec_char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, 'chroma_db_rec_char')

# 5 custom splitting
# allows creating custom splitting logic based on specific requirements
# useful for documents with unique structure that standard splitters can't handle
print(f'\n--- using custom splitter ---')

class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # custom logic for splitting text
        return text.split('\n\n') # split by paragraphs
    
custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs, 'chroma_db_custom')

# function to query a vector store
def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f'\n--- querying the vector store: {store_name} ---')
        db = Chroma(
            persist_directory=persistent_directory, embedding_function=embeddings
        )
        retriever = db.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'k':1, 'score_threshold': 0.1}
        )
        relevant_docs = retriever.invoke(query)
        # display there relevant results with metadata
        print(f'\n--- Relevant documents for {store_name} ---')
        for i, doc in enumerate(relevant_docs, 1):
            print(f'Document {i}:\n{doc.page_content}\n')
            if doc.metadata:
                print(f"source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f'vector store {store_name} does not exist.')

# define the user's question
query = 'how did Juliet die?'

# query each vector store
query_vector_store('chroma_db_char', query)
query_vector_store('chroma_db_sent', query)
query_vector_store('chroma_db_token', query)
query_vector_store('chroma_db_rec_char', query)
query_vector_store('chroma_db_custom', query)