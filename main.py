# main.py
# Kallipolis SelectorGroupChat implementation (AutoGen 0.7.5) with file logging and full transcript printing.
# Requirements:
#   pip install "autogen-agentchat>=0.7.5" "autogen-ext[ollama]>=0.7.5"
#   ollama run llama3.1:8b-instruct-q8_0
#
# Notes:
# - Writes a JSONL transcript to kallipolis_logs.jsonl.
# - Prints every turn (God, Ruler, and any specialist replies) so you can see all AgentTool dialogues.
# - Ruler prompt enforces exactly one tool call per turn with a single-argument schema: {"task": "..."}.

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.tools import AgentTool
from autogen_ext.models.ollama import OllamaChatCompletionClient

# --------------------------
# Config
# --------------------------
MODEL_NAME = "llama3.1:8b-instruct-q8_0"
TEMPERATURE = 0.6
MAX_TOOL_CALLS = 12
LOG_FILE = "kallipolis_logs.jsonl"  # persistent JSONL transcript

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def log_console(prefix: str, payload: Any) -> None:
    print(f"[{now_iso()}] {prefix}: {payload}")

def log_jsonl(speaker: str, phase: str, message: Any) -> None:
    rec = {
        "timestamp": now_iso(),
        "speaker": speaker,
        "phase": phase,
        "message": message if isinstance(message, (str, int, float, bool)) else json.dumps(message, ensure_ascii=False),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------------
# Model client
# --------------------------
def make_ollama_client() -> OllamaChatCompletionClient:
    # Keep tool schema simple for Ollama; provide model_info to ensure function_calling is enabled. [web:39]
    return OllamaChatCompletionClient(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
        },
    )

# --------------------------
# Agents
# --------------------------
def build_specialist(name: str, domain: str, client: OllamaChatCompletionClient) -> AssistantAgent:
    sys_msg = (
        f"You are {name}. Expertise: {domain}. You will receive a single TASK from the RULER. "
        "Respond in under 100 words, concise, factual, and actionable. "
        "Begin with one-line summary, followed by 1-2 short bullets. "
        "Plain text only; no JSON unless asked. "
        "All tasks are prefixed with 'FROM RULER:'."
    )
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=sys_msg,
        max_tool_iterations=1,  # specialists do not call tools
    )

def build_ruler(client: OllamaChatCompletionClient, specialist_tools: List[AgentTool]) -> AssistantAgent:
    # Enforce: exactly one tool call per turn using single-arg schema to avoid validation issues. [web:24][web:39]
    sys_msg = (
        "You are the Philosopher-Ruler: wise, rational, inquisitive, perfectionist, critical, benevolent.\n"
        "Protocol (strict):\n"
        "1) After God publishes a crisis JSON, OUTPUT a plan-of-attack with EXACT specialist NAMES (in order) and 1-2 sentence reasons each, "
        "then print single-line JSON {\"to_call\": [\"Farmer\", \"Builder\", ...]}.\n"
        "2) Then consult each listed specialist exactly once (first round) using tools with precisely one argument: "
        "{\"task\": \"FROM RULER: <task>\"}.\n"
        "   IMPORTANT: Call exactly ONE tool per turn; do NOT batch or parallelize tool calls.\n"
        "3) After all first-round replies are received, summarize and critique; if you need details, issue follow-ups using the SAME tool call format, "
        "still ONE tool call per turn; keep TOTAL tool calls ≤ 12.\n"
        "4) When satisfied, OUTPUT ONLY the final JSON (no extra text): "
        "{\"directive\": \"<single-sentence plan>\", \"called_tools\": [names], \"satisfied\": true}.\n"
        "Never send the final JSON as a tool call; use short, clear language."
    )
    return AssistantAgent(
        name="Philosopher_Ruler",
        model_client=client,
        tools=specialist_tools,
        max_tool_iterations=MAX_TOOL_CALLS,  # budget guard at the Ruler level
        system_message=sys_msg,
        description="Ruler who orchestrates consultations and outputs final directive as JSON.",
    )

def build_god(client: OllamaChatCompletionClient) -> AssistantAgent:
    sys_msg = (
        "You are God: impartial, concise, omniscient for this simulation.\n"
        "When asked to produce a crisis, reply STRICT JSON: {\"crisis\": \"...\", \"time_limit_years\": <int>}.\n"
        "When given a final directive JSON, reply STRICT JSON: {\"solved\": true/false, \"reason\": \"short\"}."
    )
    return AssistantAgent(
        name="God",
        model_client=client,
        system_message=sys_msg,
        max_tool_iterations=1,
    )

def build_agents_and_tools(client: OllamaChatCompletionClient):
    specs_meta = [
        ("Farmer", "agriculture, crops, soil and irrigation"),
        ("Builder", "infrastructure, water systems, storage, roads"),
        ("Warrior", "defense, law & order, thieves and security"),
        ("Merchant", "supply chains, storage logistics, trade"),
        ("Artist", "social morale, festivals, community cohesion"),
        ("Healer", "public health and immediate medical response"),
        ("Teacher", "education, training, public outreach"),
    ]
    specialists = {n: build_specialist(n, d, client) for (n, d) in specs_meta}

    # Wrap each specialist as an AgentTool with a single 'task': str argument for robust schema. [web:24][web:39]
    specialist_tools: List[AgentTool] = []
    for name in specialists:
        specialist_tools.append(AgentTool(specialists[name], return_value_as_last_message=True))

    ruler = build_ruler(client, specialist_tools)
    god = build_god(client)
    return god, ruler, specialists

# --------------------------
# Selector logic
# --------------------------
def _contains(text: str | None, token: str) -> bool:
    return bool(text and token in text)

def _history_flags(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> Dict[str, bool]:
    crisis_seen = any((m.source == "God") and _contains(m.to_text(), "\"crisis\"") for m in messages)
    directive_seen = any((m.source == "Philosopher_Ruler") and _contains(m.to_text(), "\"directive\"") for m in messages)
    judgement_seen = any((m.source == "God") and _contains(m.to_text(), "\"solved\"") for m in messages)
    return {"crisis": crisis_seen, "directive": directive_seen, "judgement": judgement_seen}

def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    # God -> Ruler (consult/tools/synthesize) -> God (final) deterministic handoff. [web:3]
    flags = _history_flags(messages)
    if not flags["crisis"]:
        return "God"
    if flags["directive"] and not flags["judgement"]:
        return "God"
    return "Philosopher_Ruler"

# --------------------------
# Transcript helpers
# --------------------------
def write_message_jsonl(msg: BaseChatMessage | BaseAgentEvent) -> None:
    # Persist raw text plus minimal meta for later inspection. [web:3]
    payload = {
        "source": getattr(msg, "source", None),
        "type": getattr(msg, "type", None),
        "text": msg.to_text() if hasattr(msg, "to_text") else None,
    }
    log_jsonl(payload.get("source") or "UNKNOWN", payload.get("type") or "message", payload)

def print_transcript(messages: List[BaseChatMessage | BaseAgentEvent], specialists: Dict[str, AssistantAgent]) -> List[str]:
    # Print all messages in order and return the list of consulted specialists. [web:3]
    consulted = []
    spec_names = set(specialists.keys())
    print("\n--- FULL TRANSCRIPT ---")
    for m in messages:
        src = getattr(m, "source", "UNKNOWN")
        txt = m.to_text() or ""
        # Write JSONL record for every message
        write_message_jsonl(m)
        # Console pretty-print
        print(f"\n[{src}]")
        print(txt.strip() or "<no text>")
        # Track consulted specialists by appearance
        if src in spec_names and src not in consulted:
            consulted.append(src)
    print("\n--- END TRANSCRIPT ---\n")
    return consulted

# --------------------------
# Entrypoint
# --------------------------
async def main() -> None:
    # Reset log file each run
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    client = make_ollama_client()
    god, ruler, specialists = build_agents_and_tools(client)

    termination = TextMentionTermination("solved", sources=["God"]) | MaxMessageTermination(80)  # safety cap [web:3]
    team = SelectorGroupChat(
        [god, ruler],
        model_client=client,
        selector_func=selector_func,     # deterministic next-speaker choice [web:3]
        allow_repeated_speaker=True,     # Ruler will speak repeatedly [web:3]
        termination_condition=termination,
        model_client_streaming=False,
    )

    task = (
        "Simulation start.\n"
        "God: Produce a STRICT crisis JSON now.\n"
        "Ruler: After God posts the crisis JSON, follow your protocol (plan, consult each tool exactly once in round 1, synthesize, finalize).\n"
        "God: After the Ruler emits final directive JSON, judge with STRICT {\"solved\":..., \"reason\":...}."
    )

    log_console("RUN", "Starting team...")
    run_result = await team.run(task=task)
    log_console("RUN", f"Stop reason: {run_result.stop_reason}")

    # Print and persist full transcript (includes specialist dialogues if any)
    consulted = print_transcript(run_result.messages, specialists)

    # Extract crisis, directive, and judgement best-effort for a compact summary
    crisis = None
    directive = None
    judgement = None
    for m in run_result.messages:
        src = getattr(m, "source", None)
        txt = m.to_text() or ""
        if src == "God" and '"crisis"' in txt:
            crisis = txt
        if src == "Philosopher_Ruler" and '"directive"' in txt:
            directive = txt
        if src == "God" and '"solved"' in txt:
            judgement = txt

    print("\n--- SUMMARY ---")
    print("CRISIS:", crisis or "N/A")
    print("DIRECTIVE:", directive or "N/A")
    print("CALLED_TOOLS:", consulted or [])
    print("GOD_JUDGEMENT:", judgement or "N/A")

if __name__ == "__main__":
    asyncio.run(main())
