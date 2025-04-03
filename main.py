import pandas as pd
from crewai import Crew
from crewai import CrewOutput
from agents.transaction_scanner import create_transaction_scanner_agent, create_transaction_scanner_task
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

def main():
    # Load one sample transaction
    df = pd.read_csv("data/transactions.csv")
    transaction = df.iloc[4].to_dict()

    # Create agent and task
    agent = create_transaction_scanner_agent()
    task = create_transaction_scanner_task(transaction, agent)

    # Build and run the crew
    crew = Crew(
        agents=[agent],
        tasks=[task]
    )
    result = crew.kickoff()
    # Parse the result as JSON if it's a JSON string
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            print("Result is not a valid JSON string.")
    elif isinstance(result, dict):
        print("Result is already a dictionary.")
    elif isinstance(result, CrewOutput):
        result = result.to_dict()  # Convert CrewOutput to a dictionary
        print("Converted CrewOutput to dictionary.")
    else:
        print("Unexpected result type:", type(result))
    
    print("=== CrewAI Result ===")
    print(result)

if __name__ == "__main__":
    main()