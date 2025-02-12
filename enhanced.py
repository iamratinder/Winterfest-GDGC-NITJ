import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from difflib import get_close_matches
import time
from typing import Dict, List, Optional

class HistoryChatbot:
    def __init__(self, api_key: str, data_file: str):
        """Initialize the chatbot with API key and data file."""
        os.environ["GOOGLE_API_KEY"] = api_key
        self.chat_model = ChatGoogleGenerativeAI(model="gemini-pro")
        self.historical_data = self._load_data(data_file)
        self.setup_prompt_chain()
        
    def _load_data(self, data_file: str) -> List[Dict]:
        """Load and validate historical data from JSON file."""
        try:
            with open(data_file, "r", encoding='utf-8') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError("Data must be a list of historical events")
                return data
        except FileNotFoundError:
            print(f"Error: Could not find {data_file}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {data_file}")
            return []

    def setup_prompt_chain(self):
        """Set up the prompt template and chain."""
        self.prompt = PromptTemplate(
            input_variables=["event", "year", "description", "key_figures", "question"],
            template="""
You are a passionate historian sharing knowledge about:

📅 Event: {event}
📆 Year: {year}
📚 Description: {description}
👥 Key Figures: {key_figures}

Please address this question with historical insight and engaging details:
{question}

Remember to:
- Highlight interesting connections to other historical events
- Share fascinating lesser-known facts
- Explain the historical significance
- Make the output short and concise
- don't give output in markdown format just give like as it looks beautiful and great in terminal window
"""
        )
        self.chain = LLMChain(llm=self.chat_model, prompt=self.prompt)

    def retrieve_history(self, query: str) -> Optional[Dict]:
        """Find the most relevant historical event using fuzzy matching."""
        if not self.historical_data:
            return None
            
        # Create a searchable index of event names and descriptions
        search_corpus = [
            (entry["event"].lower(), entry) for entry in self.historical_data
        ] + [
            (entry.get("description", "").lower()[:50], entry) for entry in self.historical_data
        ]
        
        # Search in both event names and descriptions
        for text, _ in search_corpus:
            matches = get_close_matches(query.lower(), [text], n=1, cutoff=0.5)
            if matches:
                for search_text, entry in search_corpus:
                    if search_text == matches[0]:
                        return entry
        return None

    def format_response(self, history: Dict) -> Dict:
        """Format the historical data for the prompt."""
        return {
            "event": history.get("event", "Unknown Event"),
            "year": history.get("year", "Unknown Year"),
            "description": history.get("description", "No Description Available"),
            "key_figures": ", ".join(history.get("key_figures", [])),
        }

    def run(self):
        """Run the interactive chat session."""
        print("\n🎓 Welcome to the Historical Knowledge Explorer! 🌟")
        print("Ask me about any historical event, or type 'exit' to quit.")
        print("Type 'help' for additional commands.\n")

        while True:
            try:
                user_input = input("\n🤔 You: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\n👋 Thank you for exploring history with me! Goodbye!")
                    break
                    
                if user_input.lower() == "help":
                    print("\n📚 Available commands:")
                    print("- 'list': Show available historical events")
                    print("- 'help': Show this help message")
                    print("- 'exit' or 'quit': End the session")
                    continue
                    
                if user_input.lower() == "list":
                    print("\n📜 Available historical events:")
                    for event in self.historical_data:
                        print(f"- {event['event']} ({event['year']})")
                    continue

                print("\n🔍 Searching for relevant historical information...")
                history = self.retrieve_history(user_input)

                if history:
                    print("💡 Found relevant historical information!")
                    response_data = self.format_response(history)
                    response_data["question"] = user_input
                    
                    # Add loading animation
                    for _ in range(3):
                        print(".", end="", flush=True)
                        time.sleep(0.3)
                    print("\n")
                    
                    response = self.chain.invoke(response_data)
                    print("\n🎯 AI Historian:", response["text"])
                else:
                    print("\n❌ I couldn't find specific information about that event in my database.")
                    print("💡 Try asking about a different historical event or type 'list' to see available events.")

            except Exception as e:
                print(f"\n⚠️ An error occurred: {str(e)}")
                print("Please try again with a different question.")

def main():
    """Main function to run the chatbot."""
    # Put your credentials and enjoy
    # API_KEY   # Replace with your actual API key
    # DATA_FILE = "history_data.json"
    
    # chatbot = HistoryChatbot(API_KEY, DATA_FILE)
    # chatbot.run()

if __name__ == "__main__":
    main()