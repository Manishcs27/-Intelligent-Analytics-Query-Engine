# Intelligent Analytics Query Engine

A production-minded prototype that converts natural-language business questions into a **validated analytical plan**, executes that plan against CSV data with pandas, and returns a JSON answer with confidence and explanation.

## Architecture

```text
Natural language query
        |
        v
+---------------------+
| GenAI Query Planner |  <-- Google Gemini API
+---------------------+
        |
        v
Structured JSON DSL
        |
        v
Schema / semantic validation
        |
        v
Deterministic pandas executor
        |
        +---- targets.csv join/comparison
        |
        v
Result + confidence + explanation
```

### Why this design?

The LLM does **not** generate arbitrary Python and execute it. Instead, it produces a small analytical DSL containing fields such as `metric`, `group_by`, `filters`, `rank`, and comparison operations.

The executor validates the generated plan before running the requested operation. This makes the system safer, easier to test, and reduces the risk of hallucinated calculations.

The GenAI component has a meaningful role: it maps business language such as "sales", "AOV", "missed target", "top 3 per region", and "YoY" to the canonical schema and supported analytical operations.

## Supported Analytical Operations

- Aggregations: sum, count, average, min, max
- Grouping
- Filtering
- Top/bottom N
- Top N within groups
- Contribution percentages
- Target comparisons using `targets.csv`
- Year-over-year growth
- Time filters such as month/year
- Nested customer ranking
- Business synonyms from `data_dictionary.json`

## Input Data

Place the supplied files under `data/`:

- `sales_data.csv`
- `targets.csv`
- `data_dictionary.json`
- `nl_queries.json`
- Optional `feedback_log.csv`

The CSV loader supports both ordinary CSV files and the quoted-row format used by the supplied dataset.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## GenAI Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=your_available_gemini_model
```

The API key should never be committed to GitHub.

## Running the Application

Run the application:

```bash
python app.py
```

Or provide a natural-language query directly:

```bash
python app.py --query "Top 2 cities by profit"
```

Example queries:

```text
Top 2 cities by profit
```

```text
Top 3 products by revenue
```

```text
Which city had the highest profit?
```

```text
What percentage of total revenue came from New York?
```

```text
Compare revenue against target by city
```

```text
What was the revenue growth year over year?
```

## Batch Evaluation

Run the supplied evaluation queries:

```bash
python app.py --batch data/nl_queries.json
```

## Output

Each query returns a JSON response containing the original query, generated analytical logic, result, confidence score, and explanation.

Example:

```json
{
  "query": "Top 2 cities by profit",
  "generated_logic": "...",
  "result": "...",
  "confidence_score": 0.9,
  "explanation": "..."
}
```

## Confidence Score

The confidence score is not simply an LLM-generated guess. It is calculated using deterministic signals such as:

- Semantic mapping coverage
- Whether requested fields exist
- Whether requested operations are supported
- Ambiguity penalties
- Executor validation
- Optional feedback history

A successfully executed query can still receive a lower confidence score when the natural-language request contains ambiguous language.

## Feedback Loop

If `data/feedback_log.csv` exists, the planner can use previous query/answer feedback and inject relevant examples into the planning prompt.

This allows the GenAI planner to improve preferred interpretations without changing the deterministic executor.

Expected columns can include:

```text
query,feedback,correct,notes
```

## Tradeoffs

### Pandas Instead of SQL

Pandas keeps the prototype dependency-light and makes grouping and nested Top-N operations straightforward.

For very large datasets, a SQL or analytical database backend would be more appropriate.

### LLM Planning Instead of Direct Text-to-SQL

The analytical DSL prevents arbitrary code execution and allows every requested field and operation to be validated before execution.

This provides a controlled boundary between GenAI and deterministic computation.

### Deterministic Fallback

When the Gemini API is unavailable, the project can fall back to a rule-based planner for supported evaluation queries.

For unseen natural-language queries, configuring the Gemini API is recommended.

## Edge Cases Considered

The engine considers several common analytical edge cases:

- Empty result sets
- Division by zero in percentages, AOV, and YoY calculations
- Missing previous-year values
- Unknown dimensions and metrics
- Top N larger than the number of available groups
- Ties in ranking
- Target rows with no matching sales
- Duplicate customer/order aggregation
- Malformed or quoted-row CSV files

## Project Structure

```text
intelligent_analytics_query_engine/
│
├── data/
│   ├── sales_data.csv
│   ├── targets.csv
│   ├── data_dictionary.json
│   ├── nl_queries.json
│   └── feedback_log.csv
│
├── app.py
├── test_engine.py
├── requirements.txt
├── sample_outputs.json
├── README.md
├── .env
├── .env.example
└── .gitignore
```

## Technologies Used

- Python
- Google Gemini API
- Google GenAI Python SDK
- Pandas
- python-dotenv
- JSON
- CSV

## Security

The system follows a separation between AI-based interpretation and actual data execution:

```text
Natural Language Query
        |
        v
      Gemini
        |
        v
Structured Analytical Plan
        |
        v
Plan Validation
        |
        v
Pandas Execution
        |
        v
Analytical Result
```

The AI model does not directly execute arbitrary Python code against the dataset.

The Gemini API key is stored in `.env` and should never be committed to the repository.

## Sample Outputs

See:

```text
sample_outputs.json
```

for example responses generated by the engine.

## Improvements With More Time

1. Add a richer semantic catalog and embedding-based schema retrieval.
2. Add a SQL backend such as DuckDB for larger datasets.
3. Add automatic query-plan verification with a second LLM pass.
4. Add a broader test suite with generated adversarial queries.
5. Store feedback in a vector store and retrieve semantically similar corrections.
6. Add a web-based UI and query history.
7. Add interactive charts and visualization recommendations.
8. Add support for conversational follow-up questions.
9. Add CSV and Excel export functionality.

## Future Vision

The long-term goal is to build a conversational business intelligence system where users can interact with business data using natural language without needing to write SQL or Python.

Example:

```text
User:
Show me the top 3 cities by revenue.

Engine:
[Results]

User:
Now compare them with their targets.

Engine:
[Target comparison]

User:
Which city performed best?

Engine:
[Analysis]
```

## License

This project is developed for educational and demonstration purposes.

## Author

Developed as an **Intelligent Analytics Query Engine** demonstrating the integration of:

**Generative AI + Natural Language Understanding + Data Analytics + Deterministic Execution**
