import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'books', 'odyssey.txt')
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("persistent directory doesn't exist. Initializing vector store...")

    # ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'the file {file_path} does not exists. please check the path'
        )
    
    # read the text content from the document file
    loader = TextLoader(file_path)
    documents = loader.load()

    # split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # display information about the split documents
    print(f'\n--- Document Chunks Information ---')
    print(f'number of sample chunks: {len(docs)}')
    print(f'sample chunk:\n{docs[0].page_content}\n')

    # create embeddings
    print('--- load embeddings function ---')
    embeddings_ef = OpenAIEmbeddings(
        model='text-embedding-3-small'
    )

    # create the vectore store and persist it automatically
    print('--- creating vector store ---')
    db = Chroma.from_documents(
        docs, embeddings_ef, persist_directory=persistent_directory
    )
    print('--- finished creating vector store ---')

else:
    print('vector store already exists. no need to initialize')