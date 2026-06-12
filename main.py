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

async def demo_batch_execution():
    """Demonstrated batch execution for multiple inputs"""
    prompt = ChatPromptTemplate.from_template("Translate to french: {text}. respond only with the translated text")
    model = ChatOpenAI(model='gpt-4o-mini' , temperature=0.5 )
    parser = StrOutputParser()
    chain = prompt | model | parser 
    inputs = [{'text':text} for text in [
        "How are you?" , 
        "What's your name?" , 
        "Where is the nearest restaurant?" ,
    ]]
    results = chain.batch(inputs)    
    for input , result in zip(inputs, results) :
        print(f"Input: {input['text']} => Output: {result}")


async def demo_streaming() :
    """Demonstrates steaming for real time output"""
    prompt = ChatPromptTemplate.from_template("Tell me a story about: {topic}")
    model = ChatOpenAI(model='gpt-4o-mini' , temperature=0.7)
    parser = StrOutputParser()
    chain = prompt | model | parser 
    print('Streaming output: ')
    for chunk in chain.stream({'topic':"Steer attention + Do Well + Be Kind = Shape Your life"}):
        print(chunk,end='' , flush=True)
        # print('')


async def schema_inspection() :
    """ Demonstrates input/output schema inspection """
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-4o-mini" , temperature=0.7)
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()
    
    
    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")

if __name__ == '__main__' :
    # asyncio.run(demo_basic_chain())
    # asyncio.run(demo_batch_execution())
    # asyncio.run(demo_streaming())
    asyncio.run(schema_inspection())