# RevenueOps Campaign Insights API

A small Python API that turns campaign performance CSVs into validated, cleaned metrics and explainable anomaly signals. It is designed as a portfolio project for data processing and AI enabled Revenue Operations workflows.

## What it does

- Validates campaign rows and numeric metrics with Pydantic.
- Cleans duplicate rows and missing campaign labels with pandas.
- Calculates spend, impressions, clicks, conversions, CTR, CVR, CPC, and CPA.
- Flags unusually high or low conversion counts using an Isolation Forest model when scikit-learn is installed and enough campaign rows are available; otherwise it uses an explainable IQR fallback.
- Produces concise, rule based follow up suggestions.
- Offers an optional OpenAI compatible LLM summary endpoint. It is disabled unless `OPENAI_API_KEY` and `OPENAI_MODEL` are configured.
- Includes synthetic CSV data so it can be run without credentials or private data.

The anomaly model is a small-data demonstration, not a production advertising recommendation system. The LLM feature sends only the aggregate analysis supplied to that endpoint and should be used with non-sensitive data unless your organization has approved the provider and data handling.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API interface.

## Analyze the sample file

```bash
curl -X POST http://127.0.0.1:8000/analyze-csv \
  -F "file=@data/sample_campaigns.csv"
```

## API routes

| Route | Purpose |
|---|---|
| `GET /health` | Basic service health |
| `POST /analyze` | Analyze a JSON list of campaign rows |
| `POST /analyze-csv` | Upload and analyze a CSV file |
| `POST /summarize` | Optional LLM narrative for aggregate analysis |

### JSON example

```json
{
  "campaigns": [
    {"campaign": "Spring Search", "spend": 120.0, "impressions": 9000, "clicks": 310, "conversions": 22},
    {"campaign": "Brand Retargeting", "spend": 90.0, "impressions": 5400, "clicks": 270, "conversions": 18}
  ]
}
```

## Optional LLM summary

Set `OPENAI_API_KEY` and `OPENAI_MODEL` to use an OpenAI compatible chat completions endpoint. `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`. No key is required for any other route. Do not put secrets in source control.

## Project structure

```text
app/
  main.py          FastAPI routes and request validation
  analytics.py     Cleaning, KPIs, anomaly detection, suggestions
  llm_summary.py   Optional API-compatible summary adapter
data/
  sample_campaigns.csv
```

## Limitations

- The sample data is synthetic.
- Isolation Forest needs at least eight distinct campaign rows and scikit-learn; smaller files or environments without scikit-learn use an IQR rule.
- Ratios with a zero denominator are returned as `null`.
- Follow up suggestions are transparent heuristics and should be reviewed by a person.
- This project does not claim a measured business uplift or production deployment.

## License

MIT
