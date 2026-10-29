# AI CSV Analyst

A local-first CSV analysis dashboard. Upload a CSV, ask a plain-English question, let a local Ollama model translate it into a constrained JSON plan, and let Pandas perform the calculation. Plotly renders interactive charts; Streamlit provides the UI.

## Safety design

- The LLM **does not generate or execute Python code**.
- Its output must be a JSON object with a small allow-listed operation: `summary`, `aggregate`, `trend`, or `anomalies`.
- The app validates column names and operation arguments before execution.
- Calculations run through fixed Pandas functions in `execute_plan`.
- No paid API key is needed. Ollama runs the language model locally.

## Requirements

- Python 3.10+
- Ollama: https://ollama.com/
- Enough RAM for your chosen local model. `qwen2.5:3b` is a lightweight starting point.

## Run on Linux (including Zorin OS)

```bash
# 1. Install Ollama using the official instructions:
#    https://ollama.com/download

# 2. Download a local model
ollama pull qwen2.5:3b

# 3. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install app dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Start the dashboard
streamlit run app.py
```

Ollama usually listens at `http://localhost:11434`. If it is not running automatically, start it in another terminal with:

```bash
ollama serve
```

Open the local Streamlit URL printed in your terminal (usually `http://localhost:8501`).

## Supported questions

- “Give me summary statistics for the dataset.”
- “Which region has the highest total sales?”
- “Show monthly revenue trend.”
- “Find unusual transaction amounts.”
- “Count orders by category.”

The dataset should have clear column names. For trends, use a date-like column and a numeric value column. For aggregates, the grouping column and value column must exist in the CSV.

## Notes and limitations

- The trend operation currently groups by each distinct date value; for daily/monthly data, pre-format dates or extend the app with resampling.
- Anomaly detection uses the standard 1.5×IQR rule. It flags candidates, not definitive errors.
- The first version intentionally keeps the operation set small and auditable. It does not support arbitrary multi-step questions or joins.
- For privacy, do not upload data you are not authorized to process. With local Ollama, prompts are sent to your local Ollama server rather than a hosted LLM API.


## Deploy to Streamlit Community Cloud (no paid API)

This app now has a dependency-free fallback planner. If Ollama is unreachable, it uses a small, explicit keyword/column matching function to choose one of the same allow-listed plans. Pandas still performs the calculations; no generated Python is executed.

1. Push the project files to a **public GitHub repository**.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud installs packages from `requirements.txt`.
5. In the sidebar, Ollama URL/model settings are optional. A cloud deployment normally cannot reach Ollama running on your personal computer, so the fallback planner is used unless you provide a reachable Ollama server.

### Important trade-off

The built-in fallback is free and runs in Streamlit Cloud, but it is a lightweight heuristic—not an LLM. It handles common questions such as summaries, group totals/averages/counts, trends, and anomalies. More nuanced language understanding requires a reachable model endpoint; public free endpoints can have quotas, latency, and changing availability.
