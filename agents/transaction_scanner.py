from crewai import Agent, Task
from crewai import LLM
from textwrap import dedent
import os
from pydantic import BaseModel

class RiskAssessment(BaseModel):
    risk_level: str
    reason: str    

llm = LLM(
    model="gemini-2.0-flash",
    base_url="https://api.openai.com",
    api_key=os.getenv("GEMINI_API_KEY")
)

def create_transaction_scanner_agent():
    return Agent(
        role="Transaction Scanner",
        goal="Classify transaction risk level and explain",
        backstory="Expert in fraud analysis and transaction compliance",
        allow_delegation=False,
        verbose=True,
        llm="gemini/gemini-2.0-flash"
    )

def create_transaction_scanner_task(transaction: dict, agent):
    prompt = dedent(f"""
        You are a financial compliance expert. Your job is to classify the risk of a financial transaction.

        Here is the transaction:

        Transaction ID: {transaction["txn_id"]}
        Customer ID: {transaction["customer_id"]}
        Amount: ₹{transaction["amount"]}
        Timestamp: {transaction["timestamp"]}
        Beneficiary Name: {transaction["beneficiary_name"]}
        Location: {transaction["location"]}

        Classify the transaction risk as:
        - LOW: Normal behavior, small or typical amount
        - MEDIUM: Slightly unusual amount or location
        - HIGH: Very large amount, foreign location, suspicious patterns

        Give your output in the format:
        {{
          "risk_level": "LOW" or "MEDIUM" or "HIGH",
          "reason": "Your reason here"
        }}
    """)

    return Task(
        description=prompt,
        agent=agent,
        expected_output="JSON with risk_level and reason", 
        output_json=RiskAssessment,
    )