from crewai import Agent, Task
from textwrap import dedent
from pydantic import BaseModel

# Output schema
class EscalationDecision(BaseModel):
    escalated: bool
    reason: str

# Agent definition
def create_escalation_manager_agent():
    return Agent(
        role="Escalation Manager",
        goal="Decide if the transaction should be escalated for review",
        backstory="Handles compliance alerts and decides whether to escalate cases based on red flags",
        allow_delegation=False,
        verbose=True,
        llm="gemini/gemini-2.0-flash"
    )

# Task definition
def create_escalation_manager_task(transaction: dict, agent, prev_results: dict):
    risk = prev_results.get("risk_level", "").upper()
    kyc = prev_results.get("kyc_status", "").upper()
    sanction = prev_results.get("sanction_match", False)

    prompt = dedent(f"""
        You are a compliance escalation officer. Based on the outputs from previous agents, decide if this transaction should be escalated.

        Transaction ID: {transaction["txn_id"]}
        Risk Level: {risk}
        KYC Status: {kyc}
        Sanction Match: {sanction}

        Escalate if:
        - Risk Level is HIGH
        - OR KYC Status is INCOMPLETE or RISKY
        - OR Sanction Match is true

        Output format:
        {{
          "escalated": true or false,
          "reason": "Explanation"
        }}
    """)

    return Task(
        description=prompt,
        agent=agent,
        expected_output="JSON with escalated and reason",
        output_json=EscalationDecision,
    )