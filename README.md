<p align="center">
  <img src="docs/lubot-logo.png" alt="LuBot" height="60">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="docs/nvidia-logo.png" alt="NVIDIA" height="70">
</p>

<h2 align="center">LuBot NVIDIA Routing</h2>

<p align="center">
  <a href="https://REPLACE_WITH_VIDEO_LINK"><img src="docs/youtube-icon.png" height="18"> <strong>Watch 4-min Demo</strong></a> &nbsp;|&nbsp;
  <a href="https://lubot.ai"><img src="docs/lubot-logo.png" height="18"> <strong>Try Live</strong></a>
</p>

<p align="center"><i>I built LuBot alone, from zero, over the last 8 months. 100% powered by NVIDIA — cloud APIs + self-hosted GPU.</i></p>

**[LuBot.ai](https://lubot.ai)** is live right now. Real business users. PhD-level statistical insights. A self-learning RAG system that gets smarter over time — finding patterns in your data that consultants charge thousands to discover.

Real math. Real statistics. Best NVIDIA.

![LuBot Home](docs/screenshots/01-lubot-home.png)

---

## **TECHNICAL INNOVATION**

### The 4-Tier Intent Classification Cascade

Most AI apps send every user message to an LLM just to figure out what the user wants. Thats slow and expensive. I built a cascade that catches 95% of queries before any LLM is needed:

```
User asks a question
        |
        v
 Intent Classifier (4 tiers - cheapest method first)
 |-- Tier 0: Keywords like "correlation", "HHI"   --> instant, 0ms, free
 |-- Tier 1: Regex patterns                        --> 80% of queries caught here
 |-- Tier 2: NVIDIA Embeddings (semantic match)    --> 5ms for ambiguous stuff
 |-- Tier 3: NVIDIA Nano 8B LLM                    --> only ~5% need this
        |
        v
 Response Tier Router
 |-- Simple ("how many employees?")     --> direct answer, skip the LLM
 |-- Medium ("revenue by region")       --> Nano 8B adds insight
 |-- PhD level ("correlation X vs Y")   --> Ultra 253B full statistical analysis
        |
        v
 LLM Router (Netflix-style failover)
 |-- NVIDIA first (every single request, always)
 |-- Groq fallback (only when NVIDIA is actually down, ~1% of time)
        |
        v
 User gets their answer
```

### Netflix-Style Failover

NVIDIA gets tried on every request. Its not "sometimes NVIDIA, sometimes Groq." NVIDIA is always primary. Groq is the safety net for infrastructure issues only. The fallback is smart - only triggers on timeouts and 503s, not on content errors that would fail the same way on Groq anyway.

### Response Tier Routing

"How many employees do we have?" doesnt need 253 billion parameters. The system detects its a simple COUNT with 1 row and returns "You have 320 employees" directly. Only complex queries like correlation analysis get routed to Ultra 253B. Use the right tool for the right job.

### Self-Learning RAG System

One of the most unique features is that LuBot has a self-learning RAG system that retrieves insights from your interactions and data patterns that you cant see by eye. The more you use it, the more it remembers about your business, your preferences, your needs. Over time LuBot becomes your best partner that knows everything about your business. Plus 22 scheduled cron jobs running daily so it never misses important data.

---

## **EFFECTIVE USE OF NVIDIA TECHNOLOGY**

| # | Technology | Model / Service | Purpose |
|---|------------|-----------------|---------|
| 1 | **Self-hosted GPU** | **NVIDIA RTX 4090** (24GB VRAM) | **Nemotron-mini** (2.7GB) + **Nemotron-3-Nano** (24GB) via Ollama |
| 2 | **Nemotron Nano 8B** | **nvidia/llama-3.1-nemotron-nano-8b-v1** | Fast classification, intent detection, simple queries |
| 3 | **Nemotron Ultra 253B** | **nvidia/llama-3.1-nemotron-ultra-253b-v1** | PhD-level statistical analysis (correlation, paradox) |
| 4 | **NV-EmbedQA-E5-v5** | **nvidia/nv-embedqa-e5-v5** | 1024-dim semantic embeddings for intent matching |
| 5 | **NIM API** | **integrate.api.nvidia.com/v1** | Cloud inference endpoint (OpenAI-compatible) |
| 6 | **AdalFlow** | **Framework** | NVIDIA-optimized LLM orchestration |
| 7 | **Nemotron-3-Nano-30B** | **nemotron-3-nano** (24GB local) | Enterprise on-premise deployment |

**100% NVIDIA Stack** — Self-hosted RTX 4090 running Nemotron locally. NIM API delivering Nano 8B and Ultra 253B from the cloud. 1024-dim embeddings for semantic matching. 7 technologies. Every layer. Every request. 99%+ success rate.

**Smart Model Routing** — 4-Tier Intent Classification + 3-Tier Response System. The right NVIDIA model for every query. Simple questions get Nano 8B. PhD analysis gets Ultra 253B. No wasted compute.

**Batched Embeddings** — 125 canonical examples pre-computed in 4 batched API calls (not 125 individual calls). Startup time: 3 seconds instead of 27.

![Correlation Analysis - PhD level powered by Ultra 253B](docs/screenshots/03-correlation-analysis.png)

![Heatmap Visualization](docs/screenshots/04-heatmap-visualization.png)

---

## **POTENTIAL IMPACT & USEFULNESS**

### The Problem

Big consulting companies and hedge funds charge thousands of dollars to look at your business data and tell you what actually matters. Most small and medium businesses cant afford that. They sit on millions of records from their customers and have no way to figure out what those numbers are really saying.

### The Solution

I built LuBot to change that.

LuBot is an AI-powered business analytics platform that helps real businesses make smarter decisions. You upload your data - CSV, Excel, whatever you have - ask questions in plain English, and get back real statistical analysis. Not just pretty charts. Real math. Correlation analysis, Simpson's Paradox detection, market concentration (HHI), anomaly detection, forecasting. The kind of stuff that a PhD data scientist at McKinsey would give you, but in language that a CEO can actually understand and act on.

LuBot is designed for profitable businesses. This is not a toy. You get real math and statistics delivered straight to you in the chat interface, or you can generate PDF reports with all the insights. You can build interactive charts, visualize your data flow, save those charts and share them with your partners.

**Try it yourself at [lubot.ai](https://lubot.ai)** - upload your data and start asking questions.

![Data Upload](docs/screenshots/02-data-upload.png)

### Privacy is Not Optional

LuBot never mixes user data. Never shares data between users. Everything is private and only you can access your data. Anytime you want you can delete everything. And if you need full control over your data - LuBot can run exclusively for your enterprise on your own infrastructure using NVIDIA Nemotron-3-Nano-30B as the local model, so nothing ever leaves your network.

---

## **QUALITY OF DOCUMENTATION & PRESENTATION**

### Live Demo

**The best way to understand LuBot is to try it: [lubot.ai](https://lubot.ai)**

Upload a CSV or Excel file, ask questions about your data, and watch it route through the NVIDIA models in real time. You'll see direct answers for simple questions, enhanced analysis for medium ones, and full PhD-level statistical breakdowns for complex queries.

### Video Walkthrough

**[📺 Watch the 4-minute demo video](https://REPLACE_WITH_VIDEO_LINK)** — Full feature demonstration

### Production Numbers

| Metric | Value |
|--------|-------|
| **Codebase** | 112,000+ lines (this repo is ~2,500) |
| **NVIDIA Success Rate** | 99%+ |
| **Response Time** | 8-10 seconds (first query), 8 seconds (warm) |
| **Database** | 34 tables, 450+ columns (Neon hot + B2 cold storage) |
| **API Endpoints** | 40+ (FastAPI) |
| **Batch Workers** | 22 daily cron jobs for learning |
| **Infrastructure** | Hetzner Cloud US, Docker, Neon PostgreSQL, Backblaze B2 |
| **Built By** | One person. 8 months. Still going. |

### Quick Start

```bash
git clone https://github.com/lubobali/LuBot-NVIDIA-AI-Agent.git
cd LuBot-NVIDIA-AI-Agent
pip install -r requirements.txt
export NVIDIA_API_KEY="nvapi-your-key-here"  # free at https://build.nvidia.com
python demo/quickstart.py
```

The demo runs through all components - intent classification, response tier routing, and NVIDIA model calls. Even without an API key the intent classifier and response router work offline.

### Code Examples

```python
from nvidia_routing import NVIDIAClient

client = NVIDIAClient()

# Simple question -> Nano 8B (fast, cheap)
response = client.chat_tier1(
    messages=[{"role": "user", "content": "Top 3 revenue drivers?"}]
)

# PhD question -> Ultra 253B (253 billion parameters)
response = client.chat_tier2(
    messages=[{"role": "user", "content": "Explain Simpsons Paradox with a business example."}]
)
```

```python
from nvidia_routing import get_llm_router

router = get_llm_router()

# NVIDIA first, always. Groq only if NVIDIA is actually down.
response = router.chat_completion(
    tier=2,
    messages=[{"role": "user", "content": "Calculate HHI for this market"}],
)
print(response.used_fallback)  # False 99% of the time
```

```python
from intent_routing import IntentClassifier

classifier = IntentClassifier()

# Tier 0 - instant deterministic detection
intent, tier, conf = classifier.classify("Whats the correlation between age and salary?")
# -> ("DATA_QUERY", 0, 1.0)
```

### Files in This Repo

```
nvidia_routing/
  llm_router.py            - Main router. NVIDIA primary, Groq fallback, smart error handling.
  nvidia_client.py         - Clean NVIDIA API wrapper. Pick a model, send messages, get response.
  nvidia_embeddings.py     - NVIDIA embeddings. SentenceTransformer-compatible drop-in.
  llm_router_client.py     - Adapter that makes the router work with AdalFlow Generator.
  response_tier_router.py  - Decides: direct answer vs enhanced vs full PhD analysis.

intent_routing/
  intent_classifier.py     - The 4-tier classification cascade. Brains of the routing.
  correlation_detector.py  - Deterministic PhD query detection. Keywords before LLM, always.

demo/
  quickstart.py            - Run this first. All components working together.
  sample_queries.py        - 15 queries showing how routing decisions get made.
```

---

[MIT License](LICENSE)

<p align="center">
  <img src="docs/nvidia-logo.png" alt="NVIDIA" height="85">
  <br><br>
  <img src="docs/lubot-logo.png" alt="LuBot" height="30">
  &nbsp;
  <strong><a href="https://lubot.ai">LuBot.ai</a></strong> — Powered by NVIDIA Nemotron
</p>
