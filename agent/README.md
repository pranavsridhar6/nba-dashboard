# NBA Data Agent (v0.1)

A LangGraph agent that answers natural-language questions about the NBA dashboard's data. Every number it reports comes from a SQL query it ran, and a deterministic check verifies that before the answer is returned.

> **Status: v0.1, CLI only.** The core agent, safety layer, grounding check and follow-up memory work. A Streamlit chat UI and a 10-question regression test set are next (see Roadmap).

## Example

```
Q: What was Boston's 2025-26 regular season record?
A: Boston went 56–26 in the 2025-26 regular season, a win percentage of .683.
   SELECT team, season, wins, losses, win_pct FROM team_season WHERE team='BOS' AND season='2025-26'

Q: And their playoff record that season?
A: Boston went 3–4 in the 2025-26 playoffs, over 7 games.
   SELECT ... FROM team_games WHERE team='BOS' AND season_type='Playoffs' ...
```

The second question names no team or season. The agent resolves "their" and "that season" from the conversation.

## Architecture

```
user question
     │
     ▼
  agent (Claude) ──wants data?──► tools (run_sql) ──► back to agent
     │
     └─ final answer ──► check (grounding) ──ok──────────────► answer
                              │
                              ├─ bad numbers, retry left ──► agent, with feedback
                              └─ still bad ──► answer + visible warning
```

| Layer | File | What it does |
|---|---|---|
| Data | `db.py` | 12 curated DuckDB views over the parquet files. The agent can only see these. Tricky logic (regular season vs playoffs) is built into the views. |
| Tool | `tools.py` | `run_sql`: one read-only SELECT, a 50-row cap, banned keywords, and errors returned as data so the model can fix its own query. Returns the rows plus the exact SQL. |
| Prompt | `prompt.py` | Scope, a reference card for the views, data facts (season formats, RAPM windows, units), and rules (every number from a query, calculations in SQL, cite the SQL). |
| Graph | `graph.py` | LangGraph state machine: agent → tools loop (ReAct), then a grounding check, with session memory per thread. |
| Check | `grounding.py` | Extracts every number from the final answer and verifies it against the tool results, allowing for rounding, $M and % formatting. No LLM involved. |

## Design decisions

- **SQL over pandas.** Queries are inspectable and easy to restrict to read-only. Running model-written pandas would mean executing arbitrary Python.
- **One general SQL tool plus curated views**, not many single-purpose tools. That covers questions nobody anticipated, while the views keep the model off raw or legacy tables.
- **Defense in depth.** The views control what is offered, and the guard controls what can run (no writes, no multiple statements, no direct file reads).
- **Deterministic grounding check.** A number comparison rather than an LLM judging an LLM. It fails closed: numbers it can't verify are flagged, never trusted.
- **Bounded loops.** A recursion limit plus exactly one grounding retry cap cost and prevent runaway calls.
- **Session memory only.** A checkpointer keyed by thread ID handles follow-up questions. Nothing persists across sessions, by design.

## Run it

```bat
pip install langgraph langchain-anthropic python-dotenv duckdb
```
Create a `.env` file in the project root (it is git-ignored):
```
ANTHROPIC_API_KEY=your-key
```
Then:
```bat
python -m agent.db          # view row counts + BOS 56-26 sanity check
python -m agent.tools       # SQL guard tests
python -m agent.grounding   # grounding check tests
python -m agent.test_check  # retry and flag paths, no API call
python -m agent.graph       # two-turn conversation
```

## Known limitations

- The grounding check verifies **numbers only**. A wrong non-numeric claim (for example, about which columns a table has) is not caught.
- Whole numbers from 0 to 10 are allowed without a source, so prose like "top 5" passes. An invented small count would also pass.
- The SQL guard uses keyword matching, so a player name containing a banned word would be refused. A real SQL parser would be stricter.
- Memory is in-process (`MemorySaver`), so it is lost when the app restarts.

## Roadmap

- [ ] Streamlit chat UI, showing the SQL and result rows behind each answer
- [ ] 10-question test set with known answers from the dashboard, auto-graded on numbers
- [ ] Recorded demo