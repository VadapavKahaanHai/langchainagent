# Financial Filing Review Assistant

A local portfolio project: one Gemini tool-calling agent with one tool, `review_filing(record_id)`. The tool uses a TF-IDF/logistic-regression classifier and returns short source excerpts, and Gemini explains the results with excerpt references.

## Setup

Python 3.11. From the project root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train, test, run

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe run.py
```

- Training writes `artifacts/filing_classifier.joblib` and evaluation files in `reports/`. Retrain if you upgrade scikit-learn.
- Tests use synthetic data and simulated model responses, so no Google credentials are needed.
- Check the real dataset/model without Gemini: `.\.venv\Scripts\python.exe -m app.tools.filing_review`
- Open http://127.0.0.1:8000/, pick a filing, choose a question, and send. Gemini requests use your API quota.

## API

Docs: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Server status |
| `GET /api/v1/agent/filings` | Record IDs |
| `GET /api/v1/agent/tools` | Tool schema |
| `POST /api/v1/agent/tools/execute/review_filing` | Local review without Gemini |
| `POST /api/v1/agent/query` | Gemini conversation and tool execution |
| `GET/DELETE /api/v1/agent/history/{session_id}` | Inspect or remove a session |

```json
{
  "query": "Review filing_001 and explain the result with excerpt references.",
  "session_id": "review-demo"
}
```

## Dataset and limitations

- 170 filings (`Fillings`, `Fraud` columns), balanced between `yes` and `no`, about 218 MB total.
- Text includes MD&A and financial statements. Sources per the dataset creator: [SEC EDGAR](https://www.sec.gov/edgar/search-and-access), [JanosAudran/financial-reports-sec](https://huggingface.co/datasets/JanosAudran/financial-reports-sec), and [SEC litigation releases](https://www.sec.gov/litigation/litreleases).
- Fraud labels were added by the creator; label rules, company grouping, and publication timing are unverified. A stratified split may include related companies.
- Results reflect performance on this small dataset, not validated real-world fraud detection. A positive prediction is not proof of fraud, and a negative one does not rule it out.
- Runs on CPU only.

## Project layout

- `train_model.py`: training and evaluation
- `app/filing_service.py`: data/model loading, prediction, excerpts
- `app/tools/filing_review.py`: the agent tool
- `app/agent/`, `app/core/llm_factory.py`: agent execution and Gemini config
- `app/api/`, `app/models/`: routes and schemas
- `app/static/`: web interface
- `tests/`: offline tests

Local single-process demo: sessions live in memory and reset on restart. No authentication or database. A report-download feature is planned (see `steps.md`).
