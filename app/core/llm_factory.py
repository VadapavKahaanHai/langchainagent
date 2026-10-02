"""Create the Gemini model used by the filing-review agent."""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings


def create_llm(
    model_name: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
) -> ChatGoogleGenerativeAI:
    key = api_key or settings.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env.")
    return ChatGoogleGenerativeAI(
        model=model_name or settings.GEMINI_MODEL,
        temperature=temperature if temperature is not None else settings.AGENT_TEMPERATURE,
        google_api_key=key,
    )
