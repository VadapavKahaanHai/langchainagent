# Financial Filing Review Assistant


A local portfolio project with one Gemini tool-calling agent and one tool, `review_filing(record_id)`. The tool uses a trained TF-IDF/logistic-regression classifier and returns short source excerpts. Gemini explains those results with excerpt references.

## Setup

Use Python 3.11 (the inspected development environment). From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
This Demo uses this dataset -



## Train, test, and run

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe run.py
```

Training writes `artifacts/filing_classifier.joblib` and evaluation files in `reports/`. Retraining overwrites those outputs. Use the same scikit-learn version when loading a saved model, or retrain after upgrading.

The automated tests use small synthetic data and simulated model responses: no Google credentials or API requests are required. To check the real local dataset/model without Gemini:

```powershell
.\.venv\Scripts\python.exe -m app.tools.filing_review
```

Open http://127.0.0.1:8000/, select a filing, choose a question, and send it. Expand the tool results to see the call and excerpts. Selecting a different filing starts a fresh conversation. Gemini requests use your account's API quota. A report-download feature remains a planned step in `steps.md`.

## API

Interactive documentation: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Server status |
| `GET /api/v1/agent/filings` | Record IDs only |
| `GET /api/v1/agent/tools` | Review-tool schema |
| `POST /api/v1/agent/tools/execute/review_filing` | Local review without Gemini |
| `POST /api/v1/agent/query` | Gemini conversation and tool execution |
| `GET /api/v1/agent/history/{session_id}` | Inspect session history |
| `DELETE /api/v1/agent/history/{session_id}` | Remove a session |

Example agent request:

```json
{
  "query": "Review filing_001 and explain the result with excerpt references.",
  "provider": "gemini",
  "agent_type": "tool_calling",
  "session_id": "review-demo"
}
```

`provider` and `agent_type` are optional and restricted to the values above. Model, temperature, and API-key overrides remain available. Alternate providers, ReAct/structured-chat modes, and the old calculator/Python/search/date tools have been removed.

## Dataset and limitations

The inspected CSV contains 170 records with `Fillings` and `Fraud` columns, split equally between `yes` and `no`. There were no blank cells or exact duplicate texts. It is approximately 218 MB; the median text length is about 85,692 words. Seven records begin with `nan`, and source text is preserved rather than blindly cleaned.

According to the creator's description supplied for this project, text includes MD&A and financial statements, with fraudulent cases relating to the year of fraud. The creator states that data was structured and labelled programmatically using Python from:

- [SEC EDGAR filings](https://www.sec.gov/edgar/search-and-access)
- [JanosAudran financial-reports-sec](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)
- [SEC litigation releases](https://www.sec.gov/litigation/litreleases)

The Hugging Face source's original labels concern stock returns; the CSV's fraud labels were added separately. Exact label-verification rules, company grouping, and publication timing have not been independently verified. A stratified record split may contain related companies. Evaluation therefore measures performance on this small supplied dataset, not validated real-world fraud detection.

The classifier predicts supplied labels; excerpts show context for associated terms and may contain boilerplate. A positive prediction is not proof of fraud, and a negative prediction does not certify absence of fraud. Training and inference run on the CPU; no GPU is required.

## Files and local state

- `train_model.py`: training, baseline comparison, evaluation, and model saving.
- `app/filing_service.py`: cached data/model loading, prediction, and excerpts.
- `app/tools/filing_review.py`: the agent tool and local smoke check.
- `app/agent/`: tool-calling execution and process-local conversation history.
- `app/core/llm_factory.py`: Gemini configuration.
- `app/api/`, `app/models/`: API routes and schemas.
- `app/static/`: HTML/CSS interface.
- `tests/`: offline regression checks.

`.gitignore` excludes secrets, virtual environments (including `langchain/`), caches, `Data/`, and `artifacts/`. Small `reports/` outputs remain available to commit for the portfolio. Ignoring a file does not remove it if it was already tracked.

This is a local, single-process demo. Conversation history and cached data live in memory; restarting clears sessions. The browser and API use the same origin. There is no authentication, database, or background worker.
