from crewai import Agent, Task
from textwrap import dedent
from pydantic import BaseModel
import sqlite3

# Output Schema
class KYCVerification(BaseModel):
    kyc_status: str
    reason: str

# Agent definition
def create_customer_verifier_agent():
    return Agent(
        role="Customer Verifier",
        goal="Verify customer's KYC status and flag risky profiles",
        backstory="Expert in identity and risk validation for compliance",
        allow_delegation=False,
        verbose=True,
        llm="gemini/gemini-2.0-flash"
    )

# Task definition
def create_customer_verifier_task(customer_id: str, agent):
    # Fetch from SQLite DB
    db_path = "data/customer_kyc.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, pan_number, aadhaar_number, address, kyc_status FROM kyc_info WHERE customer_id=?", (customer_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        kyc_data = "Customer ID not found in KYC DB"
    else:
        kyc_data = f"""
        Name: {row[0]}
        PAN: {row[1]}
        Aadhaar: {row[2]}
        Address: {row[3]}
        KYC Status: {row[4]}
        """

    prompt = dedent(f"""
        You are a KYC compliance expert. Validate the following customer's KYC status and decide if they are VALID, INCOMPLETE or RISKY.

        Customer ID: {customer_id}

        {kyc_data}

        Rules:
        - If PAN or Aadhaar is missing, mark as INCOMPLETE
        - If KYC status from DB is INCOMPLETE, reflect that
        - If everything is present and clean, mark as VALID
        - You can also mark as RISKY if address or ID looks fake

        Output format:
        {{
          "kyc_status": "VALID" or "INCOMPLETE" or "RISKY",
          "reason": "explanation here"
        }}
    """)

    return Task(
        description=prompt,
        agent=agent,
        expected_output="JSON with kyc_status and reason",
        output_json=KYCVerification,
    )