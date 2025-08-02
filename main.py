# Assingment: Build a Console-Based Support Agent System using OpenAI Agents SDK

import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunConfig, RunContextWrapper, function_tool, Handoff
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

class userInfo(BaseModel):
    name: str
    email: str
    order_id: int

@function_tool
def umfat_billing_info(user: userInfo) -> str:
    """
    Function to retrieve billing information for the user.
    """
    return f"Billing information for {user.name} with email {user.email} and order ID {user.order_id}."
@function_tool
def Umfat_technical_info(user: userInfo) -> str:
    """
    Function to retrieve technical information for the user.
    """
    return f"Technical information for {user.name} with email {user.email} and order ID {user.order_id}."
@function_tool
def Umfat_general_info(user: userInfo) -> str:
    """
    Function to retrieve general information for the user.
    """
    return f"General information for {user.name} with email {user.email} and order ID {user.order_id}."  


billing_agent = Agent(
    name="Billing Support Agent",
    instructions="You are a billing support agent. Answer questions related to billing, payments, and invoices. If you don't know the answer, say 'I don't know'.",
    tools=[umfat_billing_info],
)

technical_agent = Agent(
    name="Technical Support Agent",
    instructions="You are a technical support agent. Answer questions related to technical issues, troubleshooting, and product features. If you don't know the answer, say 'I don't know'.",
    tools=[Umfat_technical_info],
)

general_agent = Agent(
    name="General Support Agent",
    instructions="You are a general support agent. Answer general questions about the company, products, and services. If you don't know the answer, say 'I don't know'.",
    tools=[Umfat_general_info],
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You are a triage agent. Your job is to datermine the type of support needed and hand off to the appropriate agent. If you don't know the answer, say 'I don't know'.",
    handoffs=[billing_agent, technical_agent, general_agent]
)

context = userInfo(
    name="Umer Ali",
    email="umerali54544@gmail.com",
    order_id=410635
)

prompt = input("Enter your question: ")


result = Runner.run_sync(
    triage_agent,
    prompt,
    run_config=run_config,
)

print(result.final_output)