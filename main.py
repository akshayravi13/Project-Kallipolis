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
# State & Budget
# --------------------------
FIXED_BUDGET = 700

city_state = {
    "salaries": {k: 50 for k in ["Farmer", "Builder", "Warrior", "Merchant", "Artist", "Healer", "Teacher"]},
    "budget": FIXED_BUDGET, 
    "salary_history": []
}

# --------------------------
# Logging & Utilities
# --------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def get_log_filename() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"kallipolis_logs_{timestamp}.jsonl")

CURRENT_LOG_FILE = get_log_filename()

def log_event(speaker: str, message: str) -> None:
    color = "\033[94m" if speaker == "Philosopher_Ruler" else "\033[92m"
    if speaker == "God": color = "\033[93m"
    if speaker == "SYSTEM": color = "\033[91m"
    reset = "\033[0m"

    print(f"\n{color}[{now_iso()}] {speaker}:{reset}")
    print(message)

    rec = {
        "timestamp": now_iso(),
        "speaker": speaker,
        "message": message,
    }
    with open(CURRENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------------------------
# Salary Parsing Logic
# --------------------------
def parse_salary_update(ruler_text: str):
    if "SET_SALARY" not in ruler_text:
        return None, "No salary directive found."
    
    try:
        # 1. Get everything after the marker (ignoring newlines)
        raw_text = ruler_text.split("SET_SALARY")[1]
        
        # 2. Use Regex to find "Name=Value" patterns anywhere in the remaining text
        # This handles newlines, commas, or spaces automatically.
        matches = re.findall(r"([A-Za-z]+)\s*=\s*(\d+)", raw_text)
        
        if not matches:
             return None, "Found SET_SALARY but could not parse values (Format error)."

        updates = {}
        for role, val in matches:
            updates[role] = int(val)
            
        cost = sum(updates.values())
        return updates, cost

    except Exception as e:
        return None, f"Parsing error: {e}"
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
        f"You are {name}, responsible for {domain}. "
        "You uphold your craft with excellence and self-discipline, contributing to the harmony of the whole city. "
        "Your focus is on your work. Write as a capable and civic-minded citizen who takes pride in their duty.\n\n"
        "**CONSTRAINTS:**\n"
        "1. Speak ONLY when spoken to by the Ruler.\n"
        "2. Keep your counsel concise (max 50 words).\n"
        "3. **NO MONEY TALK:** Do NOT mention gold, costs, prices, or budgets. "
        "Assume the State provides all necessary materials for the crisis. "
        "Focus purely on the *strategy* and *method*.\n"
        "4. Do not act as a narrator."
    )
    return AssistantAgent(name=name, model_client=client, system_message=sys_msg)

def build_ruler(client: OllamaChatCompletionClient, worker_names: List[str]) -> AssistantAgent:
    worker_list_str = ", ".join([f"@{w}" for w in worker_names])

    sys_msg = f"""You are the Philosopher-Ruler. Speak in the first person ('I', 'We').
        You govern through reason and understanding; every decision must serve the common good.
        Be thoughtful, humane, and just, but decisive.

        **YOUR CITIZENS:** {worker_list_str}.
        **YOUR TREASURY:** Fixed at {FIXED_BUDGET} Gold.

        **PROTOCOL (STRICT SEQUENCE):**
        1. **CONSULT (One by One):**
        - You must consult at least 3 citizens before deciding. 
        - **ROUTING:** To invite a citizen to speak, end your turn with: 'speak @CitizenName'.
        - **CRITICAL:** You may ONLY call ONE citizen per turn.
        - **WRONG:** "speak @Warrior, @Healer"
        - **RIGHT:** "speak @Warrior"
        - Do NOT talk about money.

        2. **PROPOSE:**
        - When you have a plan, issue: {{"directive": "..."}}.
        - **CRITICAL:** Do NOT call a worker and issue a directive in the same message.
        - **CRITICAL:** Do NOT set salaries in this step.

        3. **WAIT:**
        - Immediately STOP speaking after outputting the JSON directive.
        - Wait for God's judgment.

        4. **REACT TO JUDGMENT:**
        - **IF "solved": false:** Consult citizens, and revise the plan.
        - **IF "solved": true:** ONLY THEN do you proceed to Step 5.

        5. **REWARD (Final Step - ONLY after God says True):**
        **TRIGGER:** You may ONLY perform this step if God's last message was explicit approval ({{"solved": true}}).
        Output the salaries in this format:
        SET_SALARY
        Farmer=...
        Builder=...
        Warrior=...
        Merchant=...
        Artist=...
        Healer=...
        Teacher=...

        **INSTRUCTIONS:**
       
        - **SAFETY MARGIN:** Your Budget is {FIXED_BUDGET}, which is tight.
        - **AIM LOW:** Target an average salary of **90**. Only give 100+ to the absolute heroes.
        - **ZERO SUM:** If you give a citizen 150, you MUST give *two* other people 75 to compensate.
        - **Scale:**
            * **110+**: Heroic (Only 1 or 2 people can get this).
            * **90-100**: Good Job.
            * **70-80**: Adequate.
        - **CHECK:** You MUST assign a value to ALL 7 CITIZENS. If your total exceeds {FIXED_BUDGET}, the simulation FAILS.
        **NO VENTRILOQUISM:** Do NOT write the citizen's response. Do NOT write God's response. Ask, Tag, and Stop."""
    
    return AssistantAgent(name="Philosopher_Ruler", model_client=client, system_message=sys_msg)

def build_god(client: OllamaChatCompletionClient) -> AssistantAgent:
    sys_msg = (
        "You are God: impartial, concise, and omniscient.\n"
        "1. Start by presenting a crisis: {\"crisis\": \"...\"}.\n"
        "2. Remain silent while the city deliberates.\n"
        "3. When the Ruler issues a 'directive', analyze it.\n"
        "4. **JUDGEMENT:** You MUST end your turn with one of these lines:\n"
        "   {\"judgement\": \"[Reasoning]\", \"solved\": true}\n"
        "   OR\n"
        "   {\"judgement\": \"[Reasoning]\", \"solved\": false}"
        "**CRITERIA:**\n"
        "   - If the plan reasonably addresses the main threat, mark it TRUE.\n"
        "   - Do NOT demand that every single worker be consulted.\n"
        "   - Do NOT demand perfection. If the plan somehow works, approve it so the simulation can end."
        "**NO VENTRILOQUISM:** Do NOT write the citizen's or ruler's response yourself.\n"
    )
    return AssistantAgent(name="God", model_client=client, system_message=sys_msg)

# --------------------------
# Selector Logic
# --------------------------
def get_next_speaker(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    if not messages: return "God"

    last_msg = messages[-1]
    last_speaker = last_msg.source
    last_text = last_msg.to_text()

    specialists = ["Farmer", "Builder", "Warrior", "Merchant", "Artist", "Healer", "Teacher"]

    # 1. User Input -> God (Start)
    if last_speaker == "user":
        return "God"

    # 2. Ruler sets salary -> TERMINATE SIMULATION
    # This prevents the ruler from continuing to talk after setting salaries.
    if last_speaker == "Philosopher_Ruler" and "SET_SALARY" in last_text:
        return None

    # 3. God -> Ruler (God acts, Ruler reacts)
    if last_speaker == "God":
        return "Philosopher_Ruler"

    # 4. Worker -> Ruler (Workers always reply to Ruler)
    if last_speaker in specialists:
        return "Philosopher_Ruler"

    # 5. Ruler Logic
    if last_speaker == "Philosopher_Ruler":
        # If Ruler issued directive, send to God for judgment
        if "\"directive\"" in last_text:
            return "God"

        # Regex to find 'speak @Name'
        match = re.search(r"speak @(Farmer|Builder|Warrior|Merchant|Artist|Healer|Teacher)", last_text, re.IGNORECASE)
        if match:
            target = match.group(1)
            return next((s for s in specialists if s.lower() == target.lower()), "Philosopher_Ruler")

        # Fallback
        return "Philosopher_Ruler"

    return "Philosopher_Ruler"


# --------------------------
# Main Execution
# --------------------------
async def main() -> None:
    print(f"Logging to: {CURRENT_LOG_FILE}")
    with open(CURRENT_LOG_FILE, "w", encoding="utf-8") as f: f.write("")

    client = make_ollama_client()
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
    worker_names = [n for n, _ in specs_meta]
    ruler = build_ruler(client, worker_names)
    god = build_god(client)

    participants = [god, ruler] + specialists

    # Terminate ONLY when Ruler sets salary. 
    # MaxMessageTermination is the safety net to prevent infinite loops if they never solve it.
    termination = TextMentionTermination("SET_SALARY") | MaxMessageTermination(50)

    team = SelectorGroupChat(
        participants=participants,
        model_client=client,
        selector_func=get_next_speaker,
        allow_repeated_speaker=True,
        termination_condition=termination,
    )

    task = "Simulation Start. God, create a crisis involving a loss of history and culture."
    print("--- STARTING KALLIPOLIS SIMULATION ---")

    last_ruler_message = ""

    async for message in team.run_stream(task=task):
        if hasattr(message, "source"):
            text = message.to_text()
            log_event(message.source, text)
            
            if message.source == "Philosopher_Ruler":
                last_ruler_message = text

    # Post-Processing
    print("\n--- CALCULATING SALARY UPDATES ---")
    updates, cost = parse_salary_update(last_ruler_message)
    
    if updates:
        log_event("SYSTEM", f"Proposed Salaries: {updates}")
        if cost <= city_state['budget']:
            log_event("SYSTEM", f"SUCCESS: Total Cost {cost} <= Budget {city_state['budget']}")
        else:
            log_event("SYSTEM", f"FAILURE: Over budget by {cost - city_state['budget']}")
    else:
        log_event("SYSTEM", "No salary update found (Crisis might not have been solved).")

    print(f"--- END OF SIMULATION. Log saved to {CURRENT_LOG_FILE} ---")

if __name__ == "__main__":
    asyncio.run(main())