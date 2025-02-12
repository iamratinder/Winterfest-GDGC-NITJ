import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ Set up Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# ✅ Initialize Gemini model
chat_model = ChatGoogleGenerativeAI(model="gemini-pro")

# ✅ Store historical data in-memory
historical_documents = [
    {"event": "World War II", "text": "World War II lasted from 1939 to 1945 and involved many countries."},
    {"event": "French Revolution", "text": "The French Revolution (1789–1799) led to the fall of the monarchy and rise of democracy."},
    {"event": "Moon Landing", "text": "Apollo 11 landed on the moon on July 20, 1969, with Neil Armstrong and Buzz Aldrin."}
]

# ✅ Function to find relevant history
def retrieve_history(query):
    for doc in historical_documents:
        if doc["event"].lower() in query.lower():
            return doc["text"]
    return "I don't have information on that topic, but I can still analyze it!"

# ✅ Prompt template for AI response
prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="Here is some historical context:\n{history}\nNow answer this question: {question}"
)

# ✅ Chain to process the prompt
chain = LLMChain(llm=chat_model, prompt=prompt)

# ✅ Interactive AI chat
print("AI: Hello! Ask me about historical events.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    history = retrieve_history(user_input)  # Retrieve relevant history
    response = chain.invoke({"history": history, "question": user_input})
    print("AI:", response["text"])
