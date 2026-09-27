"""Top-level goal router: web-search for a relevant MCP server, get consent
before installing/running it or touching credentials, use it if it fits the
goal, and fall back to the existing computer-use loop otherwise."""

import asyncio
import getpass
import json
import subprocess
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import OPENAI_API_KEY, PLANNER_MODEL
from openai import OpenAI
from websearch import web_search

client = OpenAI(api_key=OPENAI_API_KEY)

STRATEGY_SYSTEM_PROMPT = """You are deciding how to accomplish a computer-use goal. You have
one tool: web_search. Use it to find out whether an official/well-known MCP (Model Context
Protocol) server exists for the service the goal targets, and if it plausibly supports the
ACTION the goal needs (not just reading data, if the goal requires writing/creating).

MCP servers come in two shapes — check for both:
- npm-package servers you run yourself via "npx -y <package>" (stdio transport).
- servers built into a desktop app itself (e.g. Figma's Dev Mode MCP server), exposed as a
  local HTTP/SSE endpoint once the user enables it in that app's settings — these need the
  app installed, running, and sometimes a paid plan/seat, and are NOT an npm package.

Then respond with ONLY a JSON object, no prose:
{
  "strategy": "mcp_stdio" | "mcp_http" | "computer_use",
  "npx_package": "<npm package name>" or null,
  "http_url": "<default local URL, e.g. http://127.0.0.1:PORT/mcp>" or null,
  "prerequisites": "<e.g. 'Figma desktop app running, Full or Dev seat on a paid plan'>" or null,
  "requires_api_key": true|false,
  "api_key_env_var": "<ENV_VAR_NAME>" or null,
  "reason": "<one sentence>"
}

Use "computer_use" (drive the actual website/app UI directly) whenever no clearly fitting MCP
exists, or you are unsure. computer_use always works as a fallback, so prefer it over guessing."""


def decide_strategy(goal: str) -> dict:
    resp = client.responses.create(
        model=PLANNER_MODEL,
        tools=[{"type": "web_search_preview"}],
        instructions=STRATEGY_SYSTEM_PROMPT,
        input=f"Goal: {goal}",
    )
    text = resp.output_text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return json.loads(text)


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N]: ").strip().lower() == "y"
    except EOFError:
        return False


def _run_mcp_agent(goal: str, server_config: dict) -> str:
    async def _run() -> str:
        mcp_client = MultiServerMCPClient({"target": server_config})
        tools = await mcp_client.get_tools()
        print(f"[MCP] Connected. Available tools: {[t.name for t in tools]}")
        model = ChatOpenAI(model=PLANNER_MODEL, api_key=OPENAI_API_KEY)
        agent = create_react_agent(model, tools)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": goal}]},
            config={"recursion_limit": 25},
        )
        return result["messages"][-1].content

    return asyncio.run(_run())


def _get_api_key_env(decision: dict) -> dict | None:
    """Ask for an API key if the server needs one; None means "give up on MCP"."""
    if not decision.get("requires_api_key"):
        return {}
    env_var = decision.get("api_key_env_var") or "API_KEY"
    print(f"\n[MCP] This server needs an API key ({env_var}).")
    if not _confirm("Do you want to provide it now? (No = you'll log in yourself via the browser instead)"):
        return None
    key = getpass.getpass(f"Paste your {env_var} (input hidden): ").strip()
    return {env_var: key} if key else None


def run_via_mcp_stdio(goal: str, decision: dict) -> str:
    package = decision["npx_package"]
    print(f"\n[MCP] Found candidate MCP server package: {package}")
    if not _confirm(f"Install and run '{package}' via npx? This runs third-party code from npm."):
        print("[MCP] Declined — falling back to computer-use.")
        return run_via_computer_use(goal)

    env = _get_api_key_env(decision)
    if env is None:
        print("[MCP] Falling back to computer-use so you can log in yourself.")
        return run_via_computer_use(goal)

    try:
        return _run_mcp_agent(goal, {"command": "npx", "args": ["-y", package], "transport": "stdio", "env": env})
    except Exception as exc:  # noqa: BLE001 - MCP servers fail in many unpredictable ways
        print(f"[MCP] Failed ({exc!r}) — falling back to computer-use.")
        return run_via_computer_use(goal)


def run_via_mcp_http(goal: str, decision: dict) -> str:
    url = decision.get("http_url")
    prereqs = decision.get("prerequisites") or "the relevant app installed and its MCP server enabled"
    print(f"\n[MCP] Found a built-in local MCP server at {url}.")
    print(f"[MCP] Prerequisites: {prereqs}")
    if not _confirm("Do you have these set up and ready to connect?"):
        print("[MCP] Falling back to computer-use.")
        return run_via_computer_use(goal)

    env = _get_api_key_env(decision)
    if env is None:
        print("[MCP] Falling back to computer-use.")
        return run_via_computer_use(goal)

    try:
        return _run_mcp_agent(goal, {"url": url, "transport": "streamable_http", "headers": env})
    except Exception as exc:  # noqa: BLE001
        print(f"[MCP] Failed ({exc!r}) — falling back to computer-use.")
        return run_via_computer_use(goal)


def run_via_computer_use(goal: str) -> str:
    # imported lazily to avoid a circular import (main.py imports orchestrator)
    from db import get_checkpointer, init_db
    from graph import build_graph
    import uuid

    init_db()
    checkpointer = get_checkpointer()
    app = build_graph(checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run_input = {
        "goal": goal,
        "history": [],
        "response_id": None,
        "call_id": None,
        "scale": 1.0,
        "resume_reply": None,
        "status": "running",
    }
    final_state = app.invoke(run_input, config=config)
    while final_state["status"] == "awaiting_confirmation":
        confirm_text = final_state["history"][-1]["final_message"]
        print(f"\n[CONFIRMATION NEEDED] {confirm_text}")
        answer = _confirm("Allow this action?")
        reply = "approved, proceed" if answer else "not approved"
        app.update_state(config, {"resume_reply": reply, "status": "running"})
        final_state = app.invoke(None, config=config)
    return f"status={final_state['status']}"


def accomplish_goal(goal: str) -> str:
    print(f"[orchestrator] researching how to accomplish: {goal!r}")
    decision = decide_strategy(goal)
    print(f"[orchestrator] decision: {decision}")

    strategy = decision.get("strategy")
    if strategy == "mcp_stdio" and decision.get("npx_package"):
        return run_via_mcp_stdio(goal, decision)
    if strategy == "mcp_http" and decision.get("http_url"):
        return run_via_mcp_http(goal, decision)
    return run_via_computer_use(goal)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python orchestrator.py "<goal>"')
        sys.exit(1)
    print(accomplish_goal(sys.argv[1]))
