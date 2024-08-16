from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

# print('____prompt from template____')
# template = 'tell me a joke about {topic}'
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke(dict(topic='cats'))
# result = model.invoke(prompt)
# print(result.content)

print('____prompt with system and human messages____')
messages = [
    ('system', 'you are a comedian who tells jokes about {topic}.'),
    ('human', 'tell me {jokes_count} jokes.')
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke(dict(topic='lawyers', jokes_count=3))
result = model.invoke(prompt)
print(result.content)