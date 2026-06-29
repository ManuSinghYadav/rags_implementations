from pydantic import BaseModel
from typing import Literal
from rag_2.src.core.clients import openai


class QueryIntentFilter(BaseModel):
    has_filter: bool
    file_type: Literal["PDF-Markdown", "Excel"] | None = None
    cleaned_query: (
        str  # The original query stripped of filter keywords for better vector matching
    )


def extract_query_intent(query: str) -> QueryIntentFilter:
    system_prompt = (
        "You are an AI assistant that analyzes user queries to extract metadata filters for a RAG system.\n"
        "Identify if the user is explicitly or implicitly asking for a specific file format ('PDF-Markdown' or 'Excel').\n"
        "Extract the filter and provide a cleaned version of the query stripped of file-type references."
    )

    completion = openai.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        response_format=QueryIntentFilter,
        temperature=0.0,  # Keep it deterministic
    )

    return completion.choices[0].message.parsed
