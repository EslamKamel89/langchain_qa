import asyncio

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

async def demo_basic_chain(): 
    """Demonstrates a basic chain using LCEL and runnables."""
    prompt = ChatPromptTemplate.from_template("You are a helpful assistance. Answer in one sentence: {question}")
    model = ChatOpenAI(model='gpt-4o-mini' , temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser
    
    result = chain.invoke({'question':"What's langchain"})
    print('Result: ',result)
    return chain
    
    
if __name__ == '__main__' :
    asyncio.run(demo_basic_chain())