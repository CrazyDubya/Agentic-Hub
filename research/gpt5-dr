# Simulated Worlds for LLMs & AI — Field Guide
*Last updated: 2025-08-13*

This guide collects books, papers, simulators, benchmarks, frameworks, prompts, and notable projects related to **simulated worlds** used to train, evaluate, or study LLM-based agents and AI systems.

---

## Contents
- [Books & Long-Form](#books--long-form)
- [Academic Papers & Preprints](#academic-papers--preprints)
- [Simulators & Environments](#simulators--environments)
  - [Web & GUI Worlds](#web--gui-worlds)
  - [Text-Adventure Worlds](#text-adventure-worlds)
  - [Embodied / 3D Worlds](#embodied--3d-worlds)
  - [Multi-Agent & Social Simulation](#multi-agent--social-simulation)
- [Benchmarks & Evaluation Suites](#benchmarks--evaluation-suites)
- [Frameworks, Tooling & Repos](#frameworks-tooling--repos)
- [Prompt Patterns & Agent Scaffolds](#prompt-patterns--agent-scaffolds)
- [Notable Projects / Demos](#notable-projects--demos)
- [Design Notes & Gotchas](#design-notes--gotchas)

---

## Books & Long-Form
- **Reality+: Virtual Worlds and the Problems of Philosophy** — David J. Chalmers (2022). Philosophical foundation for simulated/virtual worlds and their epistemic status.
- **The Society of Mind** — Marvin Minsky (1986). Classic lens for multi-agent cognition; useful inspiration for agent societies.
- **Life 3.0** — Max Tegmark (2017). Broad AI futures; sections relevant to simulation and open-ended environments.

> These provide conceptual grounding rather than concrete engineering recipes.

---

## Academic Papers & Preprints
- **Generative Agents: Interactive Simulacra of Human Behavior** (UIST ’23). Paper and code for “Smallville,” a town inhabited by LLM-driven agents with memory and planning.  
  Paper: https://arxiv.org/abs/2304.03442 · Code: https://github.com/joonspk-research/generative_agents

- **SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents** (2023–2024). Open-ended, scenario-driven social interactions for LLMs.  
  Paper: https://arxiv.org/abs/2310.11667

- **ALFWorld: Aligning Text and Embodied Environments for Interactive Learning** (ICLR 2021). Bridges **TextWorld** plans with embodied **ALFRED** tasks.  
  Paper: https://arxiv.org/abs/2010.03768 · Site: https://alfworld.github.io

- **BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning** (2018–2019). Instruction-following in a gridworld with a “simulated human in the loop.”  
  Paper: https://arxiv.org/abs/1810.08272 · OpenReview: https://openreview.net/forum?id=rJeXCo0cYX

- **Jericho: IF Games as RL Testbeds** (2019). Framework for 30+ parser-based text adventures.  
  Paper: https://arxiv.org/abs/1909.05398

- **WebArena: A Realistic Web Environment for Building Autonomous Agents** (2023). Self-hostable websites and long-horizon tasks.  
  Paper: https://arxiv.org/abs/2307.13854 · Site: https://webarena.dev

- **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments** (2024). Realistic cross-OS GUI tasks and evals.  
  Paper: https://arxiv.org/abs/2404.07972 · Site: https://os-world.github.io

- **VisualWebArena** (2024). Extends WebArena with visually grounded tasks.  
  Paper: https://arxiv.org/abs/2401.13649

- **OSUniverse** (2025). Benchmark for multimodal GUI-navigation agents with automated validation.  
  Paper: https://arxiv.org/abs/2505.03570

- **macOSWorld** (2025) and **WorldGUI** (2025). Additional interactive GUI benchmarks.  
  macOSWorld: https://arxiv.org/abs/2506.04135 · WorldGUI: https://arxiv.org/abs/2502.08047

---

## Simulators & Environments

### Web & GUI Worlds
- **WebArena** — Self-hostable environment with e-commerce, forums, CMS, and dev sites; tasks measure end‑to‑end success.  
  Paper: https://arxiv.org/abs/2307.13854 · Site: https://webarena.dev · Code: https://github.com/web-arena-x/webarena

- **VisualWebArena** — Adds visually grounded, screenshot-dependent tasks.  
  Paper: https://arxiv.org/abs/2401.13649

- **OSWorld** — Real desktop/web apps across Ubuntu/Windows/macOS with 300+ tasks and execution-based evals.  
  Paper: https://arxiv.org/abs/2404.07972 · Site: https://os-world.github.io

- **OSUniverse** / **macOSWorld** / **WorldGUI** — Recent GUI-agent benchmarks emphasizing navigation, latency, and robustness.  
  OSUniverse: https://arxiv.org/abs/2505.03570 · macOSWorld: https://arxiv.org/abs/2506.04135 · WorldGUI: https://arxiv.org/abs/2502.08047

### Text-Adventure Worlds
- **Jericho** — 30+ classic interactive fiction games with standardized APIs.  
  Paper: https://arxiv.org/abs/1909.05398

- **BabyAI** — Gridworld instruction-following; focuses on sample efficiency and human‑in‑the‑loop teaching.  
  Paper: https://arxiv.org/abs/1810.08272 · Code: https://github.com/mila-iqia/babyai

- **TALES (2025)** — Synthetic + human-written text adventures to stress long-horizon reasoning.  
  Paper: https://arxiv.org/abs/2504.14128

### Embodied / 3D Worlds
- **ALFWorld** — Text-planned policies executed in ALFRED’s embodied environment.  
  Paper: https://arxiv.org/abs/2010.03768

- **(Related)** *MineDojo*, *Voyager*, *VirtualHome*, *BEHAVIOR* — Embodied or household simulations useful for open-ended or skill‑composition studies.  
  MineDojo: https://arxiv.org/abs/2206.08853 · Voyager: https://arxiv.org/abs/2305.16291 · VirtualHome: https://arxiv.org/abs/1806.07011 · BEHAVIOR: https://behavior.stanford.edu/

### Multi-Agent & Social Simulation
- **Generative Agents (Smallville)** — LLM towns with persistent memory, planning, and social routines.  
  Paper: https://arxiv.org/abs/2304.03442 · Code: https://github.com/joonspk-research/generative_agents

- **SOTOPIA** — Procedurally generated social scenarios; evaluates negotiation, cooperation, deception, etc.  
  Paper: https://arxiv.org/abs/2310.11667

- **AgentVerse** — Open-source platform for multi-agent simulation (classrooms, Prisoner’s Dilemma) and task-solving swarms.  
  Code & paper: https://github.com/OpenBMB/AgentVerse

---

## Benchmarks & Evaluation Suites
- **WebArena** — Realistic web tasks with pass/fail execution checks.  
  Paper: https://arxiv.org/abs/2307.13854 · Site: https://webarena.dev

- **OSWorld** (+ **OSWorld‑Human** latency study) — Execution-based evaluation across real desktop tasks; human vs. agent gap analysis.  
  OSWorld paper: https://arxiv.org/abs/2404.07972 · Site: https://os-world.github.io · OSWorld‑Human: https://arxiv.org/abs/2506.16042

- **VisualWebArena / OSUniverse / macOSWorld / WorldGUI** — 2024–2025 wave of GUI/web evals focusing on vision grounding, navigation, and reliability.  
  VisualWebArena: https://arxiv.org/abs/2401.13649 · OSUniverse: https://arxiv.org/abs/2505.03570 · macOSWorld: https://arxiv.org/abs/2506.04135 · WorldGUI: https://arxiv.org/abs/2502.08047

- **Jericho** — Standardized text-game evals; long-horizon planning and combinatorial action spaces.  
  Paper: https://arxiv.org/abs/1909.05398

- **BabyAI** — Curriculum levels for instruction-following and sample‑efficiency studies.  
  Paper: https://arxiv.org/abs/1810.08272

---

## Frameworks, Tooling & Repos
- **AgentVerse** — Multi-agent simulation & task-solving toolkit; classroom & Prisoner’s Dilemma showcases; supports local models.  
  Repo: https://github.com/OpenBMB/AgentVerse

- **Generative Agents (Smallville)** — Open-source town simulator with memory streams, daily schedules, and replay.  
  Repo: https://github.com/joonspk-research/generative_agents

*(Also relevant but not listed in depth here: PettingZoo (multi-agent env API), Melting Pot (multi-agent eval), Browser automation wrappers, OS control harnesses.)*

---

## Prompt Patterns & Agent Scaffolds
- **Role-Play Societies (e.g., CAMEL)** — Pair or population role‑play to study emergent coordination, specialization, and tool‑use.  
  Paper: https://arxiv.org/abs/2303.17760

- **Memory, Reflection, and Planning** — Patterns popularized by *Generative Agents* and follow-ups: episodic memory stores, daily/routine planners, and retrieval‑augmented self‑reflection loops.

- **Social Goals & Norms (SOTOPIA)** — Evaluate social commonsense and negotiation under scenario constraints.  
  Paper: https://arxiv.org/abs/2310.11667

---

## Notable Projects / Demos
- **Voyager** — Skill library + autotelic exploration for Minecraft; curriculum emerges from environment affordances.  
  Paper: https://arxiv.org/abs/2305.16291

- **MineDojo** — Internet-scale knowledge for open-ended agents in Minecraft.  
  Paper: https://arxiv.org/abs/2206.08853

---

## Design Notes & Gotchas
- **Reproducibility:** Prefer **self-hostable** worlds (WebArena) and **execution-based** checks (OSWorld). Version & pin the world state.
- **Long-horizon credit assignment:** Text games (Jericho/TALES) remain efficient sandboxes for planning & memory algorithms before scaling up.
- **Latency & robustness:** GUI agents are often **slow** and brittle; see OSWorld‑Human for detailed latency breakdowns. Favor **tool-use**, **state diffing**, and **restartable** plans.
- **Social evals:** Use SOTOPIA or classroom-style AgentVerse to probe norms, cooperation, deception, and bias—track emergent conventions across runs.
- **Safety:** Sandbox external tools, redact secrets in memory stores, and record full trajectories for audit.

---

### Want me to tailor this to your stack?
Tell me your target domain (web, desktop, mobile, text games, or social sim), and I’ll turn this into a hands-on build plan with runnable baselines.
