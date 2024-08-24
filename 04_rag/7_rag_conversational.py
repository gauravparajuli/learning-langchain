import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type='similarity',
    search_kwargs={'k':3}
)

llm = ChatOpenAI(model='gpt-4o-mini')

# contextualize question prompt
# this prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    'given a chat history and the latest user question'
    'which might reference context in chat history'
    'formulate a standalone question which can be understood'
    'without the chat history. do NOT answer the question, just'
    'reformulate it if needed and otherwise return it as it is.'
)

# create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# create a history aware retriever
# this uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# answer question prompt
# this system prompt helps openAI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    'You are an assistant for question-answering tasks. Use'
    'the following pieces of retrieved context to answer the'
    'question. If you dont know the answer, just say that you'
    'dont know. User three sentences maximum and keep the answer'
    'concise'
    '\n\n'
    '{context}'
)

# create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', qa_system_prompt),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ]
)

# create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# create a retrieval chain that combines history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# function to simulate a continual chat
def continual_chat():
    print('start chatting with AI! Type "exit" to end the conversation')
    chat_history = [] # collect chat history over here
    while True:
        query = input('You: ')
        if query.lower() == 'exit':
            break
        # process the user's query through the retrieval chain
        result = rag_chain.invoke(dict(input=query, chat_history=chat_history))
        # display the AI's response
        print(f"AI: {result['answer']}")
        # update the chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))

if __name__ == '__main__':
    continual_chat()