# simulator.py
import asyncio
from crisis import run_simulation

CRISES = [
    "God, create a crisis involving a plague.",
    "God, create a crisis involving a massive fire.",
    "God, create a crisis involving a loss of history and culture",
    "God, create a crisis involving an invading barbarian horde.",
    "God, create a crisis involving a catastrophic crop failure.",
    "God, create a crisis involving a deadly airborne virus.",
    "God, create a crisis involving a massive earthquake destroying bridges and roads.",
    "God, create a crisis involving a sudden devaluation of currency and trade halt.",
    "God, create a crisis involving a spread of dangerous lies and civil unrest.",
    "God, create a crisis involving the drying up of all major wells.",
    "God, create a crisis involving a wave of inexplicable depression and apathy.",
    "God, create a crisis involving the failure of all communication networks."
]

async def run_batch():
    for i, crisis in enumerate(CRISES, 1):
        print(f"\n\n==========================================")
        print(f"RUNNING SIMULATION {i}/{len(CRISES)}")
        print(f"CRISIS: {crisis}")
        print(f"==========================================\n")
        
        # Run the simulation
        await run_simulation(crisis)
        
        # Optional: Add a small delay between runs if needed for system cooldown
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(run_batch())