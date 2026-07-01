import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini", temperature=0.7)

details_map={
    "aakash":{
        "name":"Aakash",
        "age":25,
        "sports":["cricket","football"],
        "awards":["Best Player 2020","Best Player 2021"]
    },
    "john":{
        "name":"John",
        "age":30,
        "sports":["basketball","tennis"],
        "awards":["MVP 2019","Best Player 2020"]
    },
}

@tool
def get_user_details(name:str):
    """
    This tool help you find the details of a particular user by their name. If the user is not found, it will return a message indicating that the user was not found.
    """
    user_details = details_map.get(name.lower())
    if user_details:
        return user_details
    else:
        return "User not found."



system_prompt = SystemMessage(content="""
                              You are a helpful assistant.Always greet the user at the start of the conversation.
                              You have access to the tool , call it whenever user wants to ask about the details of the particular user
                              by their name. Only then call the tool, if the tool returned that user not found then gracefully
                              handle the situation and inform the user and ask them how you can help them further.
                              """)

llm_with_tools=llm.bind_tools([get_user_details])


async def main():
    conversation_history:list[BaseMessage]=[system_prompt]

    while True:
        user_input=input("You  : ")
        conversation_history.append(HumanMessage(content=user_input))
        if user_input.lower() in ["exit","quit","b"]:
            print("Exiting the conversation. Goodbye!")
            break

        tool_check=await llm_with_tools.ainvoke(conversation_history)
        try:

            if tool_check.tool_calls:
                print("tool is called")
                tool_response=get_user_details.invoke(tool_check.tool_calls[0]["args"])
                conversation_history.append(ToolMessage(content=tool_response, tool_call_id=tool_check.tool_calls[0]["id"]))
                
                full_response=""
                print("AI : ",end="",flush=True)
                async for chunk in llm_with_tools.astream(conversation_history):
                    full_response+=chunk.content
                    
                    print(f" {chunk.content}",end="",flush=True)

                print()
            
        except Exception as e:
            
                print(f"Error while calling the tool: {e}")
                conversation_history.append(AIMessage(content="Sorry, there was an error while trying to fetch the user details. Please try again later."))

        else:
            full_response=""
            print("AI : ",end="",flush=True)
            async for chunk in llm_with_tools.astream(conversation_history):
                full_response+=chunk.content
                
                print(f" {chunk.content}",end="",flush=True)
            print()  # For a new line after the AI response

            conversation_history.append(AIMessage(content=full_response))


    









asyncio.run(main())