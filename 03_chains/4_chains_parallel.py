from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

prompt_templates = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are an expert product reviewer'),
        ('human', 'list the main features of the product {product_name}.')
    ]
)

# define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'you are an expert product reviewer'),
            ('human', 'given these {features}, list the pros of these features')
        ]
    )

    return pros_template.format_prompt(features=features)

# defines cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'you are an expert product reviewer'),
            ('human', 'given these {features}, list the cons of these features')
        ]
    )

    return cons_template.format_prompt(features=features)

# combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f'Pros:\n{pros}\n\nCons:\n{cons}'

# simplify the branches with lcel
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x:analyze_cons(x)) | model | StrOutputParser()
)

# create combined chain using LCEL
chain = (
    prompt_templates
    | model
    | StrOutputParser()
    | RunnableParallel(branches={'pros': pros_branch_chain, 'cons': cons_branch_chain})
    | RunnableLambda(lambda x: print('final output', x) or combine_pros_cons(x['branches']['pros'], x['branches']['cons']))
)

# run the chain
result = chain.invoke(dict(product_name='MacBook Pro'))

print(result)