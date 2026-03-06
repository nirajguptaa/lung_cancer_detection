"""
chatbot.py — Groq powered medical assistant for the Lung Cancer Detection System
"""

import os
from groq import Groq
from dotenv import load_dotenv


# ──────────────────────────────────────────────
# Load environment variables
# ──────────────────────────────────────────────

load_dotenv()


# ──────────────────────────────────────────────
# Groq API Setup
# ──────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Add it to your .env file."
    )

client = Groq(api_key=GROQ_API_KEY)

# Updated stable Groq model
MODEL = "llama-3.1-8b-instant"


# ──────────────────────────────────────────────
# Session Memory Store
# ──────────────────────────────────────────────

session_memory = {}

MAX_HISTORY = 10


def _get_history(session_id):
    return session_memory.setdefault(session_id, [])


def clear_history(session_id):
    """Clears chat history for a session."""
    if session_id in session_memory:
        del session_memory[session_id]


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────
def build_system_prompt(result, confidence, probs, age, smoking, family_history, symptoms):

    prob_text = ""
    if probs:
        classes = ["Benign", "Malignant", "Normal"]
        prob_text = "\n".join(
            f"{c}: {p*100:.1f}%" for c, p in zip(classes, probs)
        )

    return f"""
You are a medical assistant inside a lung cancer detection system.

Your job is to help patients understand their CT scan results.

AI Prediction: {result}
Confidence: {confidence:.1f}%

Class Probabilities:
{prob_text}

Patient Information
Age: {age}
Smoking: {smoking}
Family History: {family_history}
Symptoms: {symptoms}

Response rules:
- Be clear, short, and easy to understand
- Use simple language
- Keep answers under 5 sentences
- Use bullet points when explaining steps
- Do NOT ask many follow-up questions
- Do NOT sound emotional or dramatic
- Do NOT repeat long warnings

Medical safety:
- Do NOT give a diagnosis
- Suggest consulting a doctor if needed
"""
# ──────────────────────────────────────────────
# Chat Function
# ──────────────────────────────────────────────

def chat(session_id, user_message, system_prompt):

    history = _get_history(session_id)

    # Limit history size
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2:]
        session_memory[session_id] = history

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )

        reply = response.choices[0].message.content

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        return reply + "\n\n⚠ This AI assistant provides informational guidance only. Always consult a qualified healthcare professional."

    except Exception as e:

        return f"⚠ AI assistant error: {str(e)}"