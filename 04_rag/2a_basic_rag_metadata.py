import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, 'books')
db_dir = os.path.join(current_dir, 'db')
persistent_directory = os.path.join(db_dir, 'chroma_db_with_metadata')

print(f'books directory: {books_dir}')
print(f'persistent directory: {persistent_directory}')

# check if the chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("persistent directory doesn't exist. Initializing vector store...")

    # ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f'the file {books_dir} does not exists. please check the path'
        )
    
    # list all the text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]

    # read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding='utf-8')
        book_docs = loader.load()
        for doc in book_docs:
            # add metadata to each document indicating its source
            doc.metadata = dict(source=book_file)
            documents.append(doc)

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