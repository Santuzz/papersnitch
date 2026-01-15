import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv(".env.local")

class AgentState(TypedDict):
    conversation: List[Union[HumanMessage, AIMessage]]
    task: str

def process_task(task: str) -> str:
    llm = ChatOpenAI(temperature=0)
    graph = StateGraph[AgentState](initial_state=START, terminal_state=END)

    @graph.state(START)
    def start_state(state: AgentState) -> AgentState:
        state['conversation'].append(HumanMessage(content=f"Please help me with the following task: {state['task']}"))
        response = llm(state['conversation'])
        state['conversation'].append(AIMessage(content=response.content))
        return state

    @graph.state(END)
    def end_state(state: AgentState) -> AgentState:
        return state

    initial_state: AgentState = {
        'conversation': [],
        'task': task
    }

    final_state = graph.run(initial_state=initial_state)
    return final_state['conversation'][-1].content
                
