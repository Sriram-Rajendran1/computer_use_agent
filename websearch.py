from openai import OpenAI

from config import OPENAI_API_KEY, PLANNER_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def web_search(query: str) -> str:
    """Run a web search and return a text summary with sources."""
    resp = client.responses.create(
        model=PLANNER_MODEL,
        tools=[{"type": "web_search_preview"}],
        input=query,
    )
    return resp.output_text
