"""Shared helpers for building tool-using agents on top of a chat model.

Both the Billing Agent and Account Agent follow the same shape: bind a few
tools to the LLM, let it decide which to call, execute those calls, then ask
the LLM a second time for a structured summary of what it found. This module
factors that loop out so each agent file only has to describe its own tools,
prompt, and output schema.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_TOOL_ITERATIONS = 4


def run_tool_calling_agent(llm, tools, system_prompt: str, user_message: str) -> list:
    """Run a bounded tool-calling loop and return the resulting message history."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

    for _ in range(MAX_TOOL_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)
        if not getattr(ai_message, "tool_calls", None):
            break
        for tool_call in ai_message.tool_calls:
            tool = tool_map.get(tool_call["name"])
            if tool is None:
                result = {"error": f"Unknown tool {tool_call['name']}"}
            else:
                result = tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_call["id"])
            )
    return messages


def structured_synthesis(llm, schema, synthesis_prompt: str, messages: list):
    """Ask the LLM to turn a finished tool-calling transcript into structured output."""
    transcript = "\n".join(
        f"[{message.type}] {message.content}" for message in messages if message.content
    )
    structured_llm = llm.with_structured_output(schema)
    return structured_llm.invoke(
        [SystemMessage(content=synthesis_prompt), HumanMessage(content=transcript)]
    )
