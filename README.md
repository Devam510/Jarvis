# 🤖 Jarvis — AI-Powered Desktop Assistant

> **🚧 Work in Progress** — This project is actively under development. Features are being added and refined continuously. Expect breaking changes and incomplete modules until v1.0 is released.

---

## Overview

**Jarvis** is a production-grade, voice-activated AI desktop assistant inspired by Tony Stark's J.A.R.V.I.S. Built in Python, it combines large language models, voice I/O, autonomous task execution, and a real-time GUI into a unified personal assistant that can understand, reason, and act on your desktop.

---

## ✨ Key Features (Implemented So Far)

| Feature | Status |
|---|---|
| 🎙️ Wake-word detection + Voice command pipeline | ✅ Done |
| 🧠 LLM-powered intent understanding (Ollama / OpenAI) | ✅ Done |
| 🗣️ Text-to-Speech with SSML + barge-in support | ✅ Done |
| 👁️ Screen perception & vision module | ✅ Done |
| ⚙️ Safe task execution with UAC & risk engine | ✅ Done |
| 🔁 Autonomous agent loop with rollback & checkpointing | ✅ Done |
| 🧱 Plugin architecture for extensible skills | ✅ Done |
| 📊 Real-time observability, metrics & audit ledger | ✅ Done |
| 🛡️ Dual-channel safety (voice + GUI confirmation) | ✅ Done |
| 🖥️ PySide6 GUI dashboard (Minimal / Developer / Admin) | 🔄 In Progress |
| 🗂️ Long-term behavioral memory (ChromaDB) | 🔄 In Progress |
| 🔬 Research agent & code copilot | 🔄 In Progress |

---

## 🏗️ Architecture

```
jarvis/
├── core/          # EventBus, StateStore, Watchdog
├── audio/         # Microphone input & audio processing
├── stt/           # Speech-to-Text (Whisper)
├── tts/           # Text-to-Speech (SSML, streaming)
├── perception/    # Screen capture & vision
├── cognition/     # LLM interface, context fusion
├── execution/     # Safe executor, process graph
├── autonomy/      # Autonomous agent, task planner
├── memory/        # Behavioral memory, fast search
├── risk/          # Risk engine, guardrails
├── safety/        # UAC, path validator, sandboxing
├── observability/ # Metrics, tracing, audit ledger
├── plugins/       # Extensible skill plugins
├── ui/            # PySide6 GUI dashboard
└── utils/         # Shared utilities
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) running locally **or** OpenAI API key
- Windows 10/11 (primary target OS)

### Installation

```bash
git clone https://github.com/Devam510/Jarvis.git
cd Jarvis

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Configuration

Copy the example config and fill in your values:

```bash
# Edit config.yaml with your preferred LLM backend, microphone, etc.
```

### Run

```bash
python -m jarvis
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

The test suite covers 30+ test modules across all core subsystems.

---

## 📍 Current Status

This project is **in active development**. Here's what is being worked on right now:

- 🖥️ GUI Dashboard — PySide6 interface with real-time state visualization
- 🧠 Advanced memory — Long-term behavioral learning with ChromaDB
- 🔬 Research agent — Autonomous web research and summarization
- 📱 Multi-modal perception — Improved screen understanding

---

## 🗺️ Roadmap

- [ ] v0.5 — GUI dashboard complete
- [ ] v0.6 — Long-term memory + behavioral adaptation
- [ ] v0.7 — Mobile companion app
- [ ] v1.0 — Stable public release

---

## 🤝 Contributing

This is a personal project currently developed solo. Contributions, suggestions, and bug reports are welcome once the core architecture stabilizes.

---


> *"Sometimes you gotta run before you can walk."* — Tony Stark
