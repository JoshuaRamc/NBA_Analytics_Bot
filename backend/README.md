# NBA AnalyiXpert Backend

FastAPI + LangChain (Gemini) backend that:
1) Plans how to answer an NBA stats question (`player_lookup | top_n | filter`)
2) Executes the plan locally in pandas (free, no DB)
3) Generates a concise Markdown insight from the results

## Setup

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GEMINI_API_KEY
