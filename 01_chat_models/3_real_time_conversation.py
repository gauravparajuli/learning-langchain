from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pprint import pprint

# load api key
load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

chat_history = []

system_message = SystemMessage(content='you are a helpful AI assistant')
chat_history.append(system_message)

while True:
    query = input('You: ')
    if query.lower() == 'exit':
        break

    chat_history.append(HumanMessage(content=query))

    # get ai response using the ai model
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) # add ai message

    print(f'AI: {response}')


print('____Message History____')
pprint(chat_history)