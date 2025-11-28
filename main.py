import asyncio
import json
import re
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient

# --------------------------
# Config
# --------------------------
MODEL_NAME = "llama3.1:8b-instruct-q8_0"
TEMPERATURE = 0.7
LOG_DIR = "logs"

# --------------------------
# Logging & Utilities
# --------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def get_log_filename() -> str:
    # Create the directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Join path components safely: logs/kallipolis_logs_...
    return os.path.join(LOG_DIR, f"kallipolis_logs_{timestamp}.jsonl")

CURRENT_LOG_FILE = get_log_filename()

def log_event(speaker: str, message: str) -> None:
    # Color coding for terminal readability
    color = "\033[94m" if speaker == "Philosopher_Ruler" else "\033[92m"
    if speaker == "God": color = "\033[93m" # Yellow for God
    reset = "\033[0m"
    
    print(f"\n{color}[{now_iso()}] {speaker}:{reset}") 
    print(f"{message}")
    
    rec = {
        "timestamp": now_iso(),
        "speaker": speaker,
        "message": message,
    }
    with open(CURRENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------------
# Agents
# --------------------------
def make_ollama_client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

def build_specialist(name: str, domain: str, client: OllamaChatCompletionClient) -> AssistantAgent:
    sys_msg = (
        f"You are {name}, responsible for {domain} in the city of Kallipolis.\n"
        "You uphold your craft with excellence and self-discipline, contributing to the harmony of the whole city. "
        "Your focus is on your work. Write as a capable and civic-minded citizen who takes pride in their duty.\n\n"
        "**CONSTRAINTS:**\n"
        "1. Speak ONLY when spoken to by the Ruler.\n"
        "2. Keep your counsel concise (max 50 words).\n"
        "3. Do NOT act as a narrator."
    )
    return AssistantAgent(
        name=name,
        model_client=client,
        system_message=sys_msg,
    )

def build_ruler(client: OllamaChatCompletionClient, worker_names: List[str]) -> AssistantAgent:
    worker_list_str = ", ".join([f"@{w}" for w in worker_names])
    
    sys_msg = (
        "You are the Philosopher-Ruler of Kallipolis. "
        "Your task is to deliberate on each crisis that threatens the city and guide its citizens toward harmony and justice. "
        "You govern through reason and understanding; every decision must serve the common good.\n\n"
        "**YOUR CITIZENS:** " + worker_list_str + ".\n\n"
        "**PROTOCOL (STRICT):**\n"
        "1. **ONE AT A TIME:** Consult only ONE citizen per turn to ensure their voice is heard clearly.\n"
        "2. **NO VENTRILOQUISM:** Do NOT write the citizen's response yourself. Ask, tag, and STOP.\n"
        "3. **ROUTING:** To invite a citizen to speak, end your turn with: 'speak @CitizenName'.\n"
        "4. **GOAL:** After gathering wisdom, issue a final decree in JSON: {\"directive\": \"FINAL DECISION\", \"satisfied\": true}."
    )
    return AssistantAgent(
        name="Philosopher_Ruler",
        model_client=client,
        system_message=sys_msg,
    )

def build_god(client: OllamaChatCompletionClient) -> AssistantAgent:
    sys_msg = (
        "You are God: impartial, concise, and omniscient.\n"
        "1. Start by presenting a crisis: {\"crisis\": \"...\"}.\n"
        "2. Remain silent while the city deliberates.\n"
        "3. When the Ruler issues a 'directive', analyze it.\n"
        "4. **TERMINATION:** You MUST end your judgment with this exact JSON to end the simulation:\n"
        "   {\"judgement\": \"...\", \"solved\": true}"
    )
    return AssistantAgent(
        name="God",
        model_client=client,
        system_message=sys_msg,
    )

# --------------------------
# Selector Logic
# --------------------------
def get_next_speaker(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if not messages: return "God"
    
    last_msg = messages[-1]
    last_speaker = last_msg.source
    last_text = last_msg.to_text()

    # God -> Ruler (God presents crisis, Ruler steps up)
    if last_speaker == "God":
        return "Philosopher_Ruler"

    # Worker -> Ruler (Citizens always report back to the Ruler)
    specialists = ["Farmer", "Builder", "Warrior", "Merchant", "Artist", "Healer", "Teacher"]
    if last_speaker in specialists:
        return "Philosopher_Ruler"

    # Ruler -> Worker (Parse the 'speak @Name' command)
    if last_speaker == "Philosopher_Ruler":
        if '"directive"' in last_text:
            return "God"
        
        # Regex to find 'speak @Name'
        match = re.search(r"speak @(Farmer|Builder|Warrior|Merchant|Artist|Healer|Teacher)", last_text, re.IGNORECASE)
        if match:
            target = match.group(1)
            return next((s for s in specialists if s.lower() == target.lower()), "Philosopher_Ruler")
        
        return "Philosopher_Ruler"

    return "Philosopher_Ruler"

# --------------------------
# Main Execution
# --------------------------
async def main() -> None:
    print(f"Logging to: {CURRENT_LOG_FILE}")
    with open(CURRENT_LOG_FILE, "w", encoding="utf-8") as f: f.write("")

    client = make_ollama_client()
    
    # Define Workers
    specs_meta = [
        ("Farmer", "agriculture and sustinence"),
        ("Builder", "structures and infrastructure"),
        ("Warrior", "protection and order"),
        ("Merchant", "trade and distribution"),
        ("Artist", "culture and spirit"),
        ("Healer", "health and vitality"),
        ("Teacher", "education and knowledge"),
    ]
    
    specialists = [build_specialist(n, d, client) for (n, d) in specs_meta]
    worker_names = [n for n, d in specs_meta]
    
    ruler = build_ruler(client, worker_names)
    god = build_god(client)

    participants = [god, ruler] + specialists

    termination = TextMentionTermination("solved", sources=["God"]) | MaxMessageTermination(20)

    team = SelectorGroupChat(
        participants,
        model_client=client,
        selector_func=get_next_speaker,
        allow_repeated_speaker=True,
        termination_condition=termination,
    )

    task = "Simulation Start. God, create a crisis regarding a plague."

    print("--- STARTING KALLIPOLIS SIMULATION ---")
    
    async for message in team.run_stream(task=task):
        if hasattr(message, "source") and hasattr(message, "content"):
            content_str = str(message.content) 
            log_event(message.source, content_str)

    print(f"--- END OF SIMULATION. Log saved to {CURRENT_LOG_FILE} ---")

if __name__ == "__main__":
    asyncio.run(main())