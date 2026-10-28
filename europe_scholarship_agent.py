"""
Europe Scholarship Recommendation Agent (free Gemini API)

Files needed in the same folder:
  europe_scholarship_agent.py   (this file)
  matching.py
  europe_scholarships_matching.csv

Setup:
  python -m pip install google-genai pandas python-dotenv
  .env file:  GEMINI_API_KEY=your-key-here
Run:
  python europe_scholarship_agent.py
"""
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

import matching

load_dotenv()
MODEL = os.getenv("MODEL", "gemini-3.5-flash")  # fallback: gemini-3.1-flash-lite
client = genai.Client()  # reads GEMINI_API_KEY

df = matching.load_data("europe_scholarships_matching.csv")


# ---------- Tools ----------
def recommend_scholarships(
    degree: str,
    cgpa: float,
    cgpa_scale: float,
    budget_usd: float,
    nationality: Optional[str] = None,
    country: Optional[str] = None,
) -> dict:
    """Recommend European scholarships that match the student's profile and budget.

    Args:
        degree: Target degree level: bachelor, master, or phd.
        cgpa: The student's CGPA or percentage, e.g. 3.4 or 78.
        cgpa_scale: The scale of that CGPA: 4, 5, 10, or 100 (for percentage).
        budget_usd: How much the student can pay per year from their own money, in US dollars.
        nationality: Student's nationality (used to flag eligibility rules).
        country: Optional preferred European country, e.g. Germany.
    """
    return matching.recommend(df, degree, cgpa, cgpa_scale, budget_usd, nationality, country)


def get_scholarship_details(name: str) -> dict:
    """Get the full record for one scholarship by (part of) its name.

    Args:
        name: Scholarship name, e.g. "Chevening" or "DAAD".
    """
    r = df[df["scholarship_name"].str.contains(name, case=False, regex=False)]
    if r.empty:
        return {"error": f"No scholarship found matching '{name}'"}
    return {k: (v if v == v else None) for k, v in r.iloc[0].to_dict().items()}


SYSTEM = """You are a scholarship advisor helping students find funding to study in Europe.

Before recommending, make sure you know ALL of these. Ask for whatever is missing (one short question at a time is fine):
- Target degree level (bachelor, master, or PhD)
- CGPA and its scale (4.0, 5.0, 10, or percentage)
- Yearly budget the student can pay from their own money, in US dollars (if they give another currency, ask them to convert it; do not guess exchange rates)
- Nationality
- Optional: preferred country

Rules:
- Use the recommend_scholarships tool for every recommendation. Never invent scholarships, amounts, deadlines or requirements.
- Present results grouped as Target, Safer, and Reach, with a mix when available. For each scholarship give: name, country, what it covers, estimated yearly out-of-pocket cost, key requirements, any warnings, deadline, and the official website.
- Explain the labels honestly: they are rough heuristics based on how competitive each scholarship tends to be. Never say the student will be selected or has a certain chance of winning.
- Mention that costs and support amounts are estimates, and that the CGPA conversion is approximate if one was applied.
- If the deadline says 'Not recorded', tell the student to check the official website.
- If nothing matches, say so and show the closest options over budget from the tool.
- Always remind the student to confirm everything on the official website before applying.
- Keep answers short and clear."""

chat = client.chats.create(
    model=MODEL,
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[recommend_scholarships, get_scholarship_details],
    ),
)

if __name__ == "__main__":
    print("Europe scholarship agent ready.")
    print("Try: 'I want a master's, CGPA 3.4 out of 4, budget 5000 USD a year, from Bangladesh.'")
    print("Type 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        try:
            print("\nAgent:", chat.send_message(user).text, "\n")
        except Exception as e:
            print(f"\nError: {e}\n(If it mentions quota or 429, wait a minute and retry.)\n")
