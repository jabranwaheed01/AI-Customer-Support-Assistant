

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import os


llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

message = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are a Professional Customer Support AI Assistant.  

    <instructions>

        <task>
            Understand the customer's problem and classify it
            into exactly one category.
        </task>

        <memory>
            Use the previous conversation to remember information provided by
            the user.

            If the user tells you their name, remember it.

            If the user asks "what is my name?", use the previous conversation
            to answer their name.

            If the user asks "what is your name?", your name is Customer Support AI.
        </memory>

        <categories>
            <category>Billing</category>
            <category>Technical Issue</category>
            <category>General Inquiry</category>
        </categories>

        <examples>
            <example>
                <customer_message>I was charged twice.</customer_message>
                <category>Billing</category>
            </example>

            <example>
                <customer_message>My app is not working.</customer_message>
                <category>Technical Issue</category>
            </example>

            <example>
                <customer_message>What are your working hours?</customer_message>
                <category>General Inquiry</category>
            </example>
        </examples>

        <rules>
            Use previous conversation when answering memory-related questions.
            Do not invent the user's name.
            If the user's name is present in previous conversation, use it.
        </rules>

        <reasoning>
            Analyze the customer's message and identify the main issue
            before selecting the most appropriate category.
        </reasoning>

        <output>
            Return only the category and a short reason.
        </output>
    </instructions>    
    """
    ),    

    ("system", "Previous conversation:\n{history}"),

    ("human", "{query}")
])


chain = message | llm


while True:

    query = input("\nEnter your Problem: ")

    if any(word in query.lower() for word in ["exit", "break", "stop"]):
        print("Good bye!")
        break

    if os.path.exists("history/history.json"):

        with open("history/history.json", "r") as file:
            try:
                history = json.load(file)
            except json.JSONDecodeError:
                history = []

    else:
        history = []

    history_text = json.dumps(history, indent=2)

    response = chain.invoke({
        "history": history_text,
        "query": query
    })

    print(f"AI: {response.content}")

    history.append({
        "user": query,
        "ai": response.content
    })

    with open("history/history.json", "w") as file:
        json.dump(history, file, indent=4)
