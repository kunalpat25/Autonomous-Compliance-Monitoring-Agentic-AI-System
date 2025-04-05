from crewai import Agent, Task
from crewai import LLM
from textwrap import dedent
from pydantic import BaseModel
import pandas as pd
from fuzzywuzzy import fuzz

# Output schema
class SanctionCheck(BaseModel):
    sanction_match: bool
    matched_name: str
    confidence_score: int

# Agent definition
def create_sanction_checker_agent():
    return Agent(
        role="Sanction List Checker",
        goal="Match beneficiary name against sanction lists",
        backstory="Expert in financial crime and blacklist enforcement",
        allow_delegation=False,
        verbose=True,
        llm="gemini/gemini-2.0-flash"
    )

# Task definition
def create_sanction_checker_task(transaction: dict, agent):
    beneficiary = transaction["beneficiary_name"]
    sanctions_df = pd.read_csv("data/sanction_list.csv")
    
    best_match = None
    best_score = 0

    for name in sanctions_df["sanctioned_name"]:
        score = fuzz.token_sort_ratio(beneficiary.lower(), str(name).lower())
        if score > best_score:
            best_score = score
            best_match = name

    is_match = best_score >= 85

    prompt = dedent(f"""
        You are a sanctions enforcement officer. Based on fuzzy match score, verify if a beneficiary is likely on a global sanctions list.

        Beneficiary: {beneficiary}
        Closest Match: {best_match}
        Similarity Score: {best_score}

        If score is 85 or above, mark sanction_match as true.

        Output format:
        {{
            "sanction_match": true or false,
            "matched_name": "closest matched name",
            "confidence_score": score
        }}
    """)

    return Task(
        description=prompt,
        agent=agent,
        expected_output="JSON with sanction_match, matched_name, and confidence_score",
        output_json=SanctionCheck,
    )