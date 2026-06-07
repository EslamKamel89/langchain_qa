import asyncio
from importlib.metadata import version

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

async def main(): 
    core_version = version('langchain_core')
    langgraph_version = version('langgraph')
    
    print(f"langchain core version: {core_version}")
    print(f"langgraph  version: {langgraph_version}")
    
    llm = ChatOpenAI(model='gpt-4o-mini' , temperature=0)
    response = llm.invoke('Say `setup complete` in one word')
    
    print(f"Response from openai: {response}")
    print(f"result : {response.content}")
    
    print('Setup completed')
    
if __name__ == '__main__' :
    asyncio.run(main())