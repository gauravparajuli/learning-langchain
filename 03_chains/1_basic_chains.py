from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

messages = [
    ('system', 'you are a comedian who tells jokes about {topic}.'),
    ('human', 'tell me {jokes_count} jokes.')
]

prompt_template = ChatPromptTemplate.from_messages(messages)

chain = prompt_template | model | StrOutputParser()
result = chain.invoke(dict(topic='lawyers', jokes_count=3))

print(result)