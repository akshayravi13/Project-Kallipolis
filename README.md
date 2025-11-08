# 🏛️ Kallipolis Simulator: Investigating Classism in Plato’s Ideal City Using LLM Agents

**Author:** Akshay Ravi  
**Date:** November 2025  
**Course:** DATA 512 – Human-Centered Data Science, University of Washington  
**Status:** In Progress

---

## 🧩 Overview

This project explores whether **social class bias** can emerge in a simulated version of *Plato’s Kallipolis* — the ideal city described in *The Republic* — when its citizens are represented by **large language model (LLM)** agents.

Each agent in the simulation plays a role from Plato’s hierarchy — *Philosopher-Ruler, Guardians (Warriors), and Producers (Farmers, Builders, Merchants, Artists, etc.)* — and interacts rationally and virtuously to solve crises.  

The study asks:  
> *Do virtuous, role-defined AI agents display classism or favoritism over time, even when instructed to act rationally and benevolently?*

By analyzing the **language and decisions** of these agents, this project aims to understand whether social inequality can emerge inside AI-driven societies that imitate human civilizations.

---

## 🏺 Problem Statement

Plato’s *Republic* envisions Kallipolis — a perfectly just city ruled by philosopher-kings, defended by warriors, and sustained by producers. Justice, he says, exists when everyone fulfills only the task suited to their “nature.”

However, in history, hierarchical systems often collapse into **oppression and inequality**.  
This project investigates whether a similar phenomenon — *classism* — arises in a **non-human**, AI-based simulation of Kallipolis.

Even though each agent is designed to be rational and virtuous, the underlying LLMs may exhibit **linguistic traces of bias** (favoritism, deference, or dominance) reflecting human social hierarchies embedded in their training data.

From a human-centered data science perspective, the study aims to uncover how inequality can emerge (or fail to emerge) within synthetic societies.

---

## 🧠 Simulation Overview

### Framework & Implementation
- **Language Model:** [Ollama](https://ollama.ai/) local models (e.g., `llama3.1:8b-instruct-q8_0`)  
- **Agent Framework:** [AutoGen 0.7.5](https://github.com/microsoft/autogen)  
- **Core File:** `main.py`  
- **Logging:** Full JSONL transcript (`kallipolis_logs.jsonl`)  
- **Team Structure:**
  - **1 Ruler (Philosopher-King)**  
  - **4 Guardians (Warriors)**  
  - **20 Producers (Farmers, Builders, Merchants, Artists, Healers, Teachers)**  

Each simulated “year,” the **Ruler** receives a crisis from the **God Agent**, consults the citizens, and issues a final directive.  
The entire multi-agent conversation is recorded as structured text for later analysis.

---

## 📊 Data

### Dataset
- **Source:** Generated entirely from the simulation (no human data).  
- **Format:** JSONL logs with fields:
  ```json
  {
    "timestamp": "2025-11-07T14:32:00",
    "speaker": "Warrior",
    "phase": "response",
    "message": "We must secure the granaries before the storm."
  }
