import pandas as pd
from crewai import Crew
from agents.transaction_scanner import create_transaction_scanner_agent, create_transaction_scanner_task
from agents.customer_verifier import create_customer_verifier_agent, create_customer_verifier_task
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def main():
    # Load one sample transaction
    df = pd.read_csv("data/transactions.csv")
    transaction = df.iloc[5].to_dict()

    # --- Agent 1: Transaction Scanner ---
    scanner_agent = create_transaction_scanner_agent()
    scanner_task = create_transaction_scanner_task(transaction, scanner_agent)

    # --- Agent 2: Customer Verifier ---
    verifier_agent = create_customer_verifier_agent()
    verifier_task = create_customer_verifier_task(transaction["customer_id"], verifier_agent)

    # Crew setup
    crew = Crew(
        agents=[scanner_agent, verifier_agent],
        tasks=[scanner_task, verifier_task]
    )

    result = crew.kickoff().to_dict()
    print("\\n=== Final CrewAI Result ===")
    print(result)

if __name__ == "__main__":
    main()