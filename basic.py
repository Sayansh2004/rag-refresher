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
def greet_user(user_name:str):
    """
    use this tool to greet the user by their name. It will return a greeting message.
    """

    return f"Hello {user_name}! How can I assist you today? You can ask me about the details of a particular user by their name and what else I can help you with."
 
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

tool_map={
    "greet_user":greet_user,
    "get_user_details":get_user_details
}


system_prompt = SystemMessage(content="""
                              You are a helpful assistant named "flexibot" -> as you are flexible as per user needs or you can reply out any professioanl lines if user asks w=something about your this name.Always ask user for their name and call the greet_user tool to greet them by their name.
                              You have access to two tools, one is for greeting and another is for getting user details , call it whenever user wants to ask about the details of the particular user
                              by their name. Only then call the tool, if the tool returned that user not found then gracefully
                              handle the situation and inform the user and ask them how you can help them further.
                              """)

llm_with_tools=llm.bind_tools([get_user_details,greet_user])


async def main():
    conversation_history:list[BaseMessage]=[system_prompt]

    while True:
        user_input=input("You  : ")
        conversation_history.append(HumanMessage(content=user_input))
        tool_calls=0
        if user_input.lower() in ["exit","quit","b"]:
            print("Exiting the conversation. Goodbye!")
            break

        tool_check=await llm_with_tools.ainvoke(conversation_history)
        conversation_history.append(tool_check)
        try:

            if tool_check.tool_calls:

                async def tool_call_executor(tool_name,tool_id,tool_args):
                    tool_function=tool_map.get(tool_name)
                    if tool_function:
                        tool_response=tool_function.invoke(tool_args)
                        conversation_history.append(ToolMessage(content=tool_response, tool_call_id=tool_id))
                    else:
                        print(f"Tool {tool_name} not found.")
                        conversation_history.append(AIMessage(content=f"Sorry, I couldn't find the tool named {tool_name}. Please try again."))


                for tool_call in tool_check.tool_calls:
                    tool_id=tool_call["id"]
                    tool_args=tool_call["args"]
                    tool_name=tool_call["name"]
                    tool_calls+=1
                    await tool_call_executor(tool_name,tool_id,tool_args)

                
                
        except Exception as e:
            
                print(f"Error while calling the tool: {e}")
                conversation_history.append(AIMessage(content="Sorry, there was an error while trying to fetch the user details. Please try again later."))

         
        else:
            full_response=""
            print("AI : ",end="",flush=True)
            async for chunk in llm_with_tools.astream(conversation_history):
                    full_response+=chunk.content
                    
                    print(f" {chunk.content}",end="",flush=True)
            print()  
            conversation_history.append(AIMessage(content=full_response))

        finally:
             if tool_calls>0:
                print("Tool calls made in this iteration:", tool_calls)

             else:
                print("No tool calls made in this iteration.")

    









asyncio.run(main())