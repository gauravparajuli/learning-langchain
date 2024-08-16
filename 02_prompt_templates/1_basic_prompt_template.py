from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# always use tuples when building templates if you want to specify the type of message

# # create a template
# template = 'tell me a joke about {topic}.'
# prompt_template = ChatPromptTemplate.from_template(template)

# print('____prompt from template____')
# prompt = prompt_template.invoke(dict(topic='cat'))
# print(prompt)

# # create multiple template
# template_multiple = """
# You are a helpful assistant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant:
# """
# prompt_template = ChatPromptTemplate.from_template(template_multiple)

# print('____prompt from template____')
# prompt = prompt_template.invoke(dict(adjective='funny', animal='panda'))
# print(prompt)

# prompt with system and human messages using tuple
messages = [
    ('system', 'you are a comedian who tells jokes about {topic}'),
    ('human', 'tell me {joke_count} jokes')
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke(dict(topic='lawyers', joke_count=3))
print(prompt)