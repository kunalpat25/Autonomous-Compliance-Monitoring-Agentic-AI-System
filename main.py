import pandas as pd
from agents.transaction_scanner import create_transaction_scanner_agent, create_transaction_scanner_task
from agents.customer_verifier import create_customer_verifier_agent, create_customer_verifier_task
from agents.sanction_checker import create_sanction_checker_agent, create_sanction_checker_task
from agents.escalation_manager import create_escalation_manager_agent, create_escalation_manager_task
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
    scanner_result = scanner_task.execute()
    print("\\n[Transaction Scanner Result]")
    print(scanner_result)

    # --- Agent 2: Customer Verifier ---
    verifier_agent = create_customer_verifier_agent()
    verifier_task = create_customer_verifier_task(transaction["customer_id"], verifier_agent)
    verifier_result = verifier_task.execute()
    print("\\n[Customer Verifier Result]")
    print(verifier_result)

    # --- Agent 3: Sanction Checker ---
    sanction_agent = create_sanction_checker_agent()
    sanction_task = create_sanction_checker_task(transaction, sanction_agent)
    sanction_result = sanction_task.execute()
    print("\\n[Sanction Checker Result]")
    print(sanction_result)

    # --- Agent 4: Escalation Manager ---
    escalation_agent = create_escalation_manager_agent()
    escalation_input = {
        "risk_level": scanner_result.risk_level,
        "kyc_status": verifier_result.kyc_status,
        "sanction_match": sanction_result.sanction_match
    }
    escalation_task = create_escalation_manager_task(transaction, escalation_agent, escalation_input)
    escalation_result = escalation_task.execute()
    print("\\n[Escalation Manager Result]")
    print(escalation_result)

if __name__ == "__main__":
    main()