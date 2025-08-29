import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate



load_dotenv()

class State(TypedDict): 
    text: str
    classification: str
    entities:List[str]
    summary:str

llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classification_node(state: State):
    '''Classify the text into one of the categories: News, Blog, Reasearch, or other'''

    prompt=PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:"
    )

    message=HumanMessage(content=prompt.format(text=state["text"]))
    classification=llm.invoke([message]).content.strip()
    return {"classification" : classification} 


def entity_extraction_node(state:State):
    '''Extract all the entites (Person, Organization, Location) from the text'''

    prompt=PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Return them as a comma-separated list.\n\nText: {text}\n\nEntities:"
    )
    message=HumanMessage(content=prompt.format(text=state["text"]))
    entities =llm.invoke([message]).content.strip().split(",")
    return {"entities":  entities}

def summarization_node(state: State):
    '''summarize the antire text in fe short sentences'''

    prompt=PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in five short sentences.\n\nText: {text}\n\nSummary:"
    )
    message=HumanMessage(content=prompt.format(text=state["text"]))
    summary=llm.invoke([message]).content.strip()
    return {"summary": summary}



workflow = StateGraph(State)

#add nodes to graph
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

#add edges to graph
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

#complie workflow
app=workflow.compile()


sample_text =input("Enter the Article:") 

#"""
#OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
#additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
#"""

state_input = {"text": sample_text}
result = app.invoke(state_input)

print("Classification:", result["classification"])
print("\nEntities:", result["entities"])
print("\nSummary:", result["summary"])