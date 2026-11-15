# Handover: continuing the AI Data Scientist BTP

You are picking up a project that is about one-third built. This document tells you what exists,
what is missing, and exactly what to build next, in order. Read it top to bottom once, then follow
section 6 (the build order).

---

## 1. The idea in one minute

The project is an **autonomous multi-agent AI data scientist**: the user uploads a CSV and a goal in
plain English ("identify the main factors affecting customer churn"), and the system plans the analysis,
runs it, checks its own work, and writes a report.

The one idea that makes it different from "an LLM that writes analysis":

> **A number may only appear in a report if a tool computed it and the evidence ledger recorded it.**

LLM agents decide *what to compute*. Deterministic Python tools compute it. Every result is saved in an
append-only **evidence ledger** with an ID (`E0017`). The report is a list of **claims**, each citing
evidence IDs, and a **verifier** rejects any claim whose numbers do not match its evidence. A **Critic**
adds checks that a human statistician would do (target leakage, tiny samples, imbalance, overfitting) and
sends targeted feedback to the agent that must fix the problem.

Example: if the Insight agent writes "month-to-month customers churn at 47%" but evidence `E0017`
says 42.7%, the verifier rejects the sentence and the agent must rewrite it or drop it.

Why the original prototype was not good enough (so you know what we are fixing): its agents were fixed
functions, its retry loop re-ran identical code, its recommendations were hard-coded text unrelated to
the data, and its evaluation measured nothing.

---

## 2. Setup (do this first, ~15 minutes)

```
git clone <repo>            # or copy the folder
cd agentic-ai-data-scientist
python -m venv .venv
.venv\Scripts\Activate.ps1                      # Windows PowerShell
python -m pip install -r requirements-dev.txt
python -m pytest                                # expect everything green (~1 minute)
python scripts/smoke_tools.py                   # every tool on every real dataset
```

LLM key (free): get a Gemini key from Google AI Studio (aistudio.google.com), copy `.env.example` to
`.env`, and set `GEMINI_API_KEY=...`. **Use your own key; never commit `.env`.** Then:

```
python scripts/check_llm.py                     # proves live calls + tool-calling work
```

With no key at all the system runs in **offline mode** (a fixed tool order, no LLM). Offline mode must
always keep working: it is what makes the demo impossible to break.

Windows gotcha we hit: if `pip` crashes with `PermissionError ... sslkeys.log`, delete the
`SSLKEYLOGFILE` environment variable (user variables) and open a new terminal.

---

## 3. What is already built (and tested)

| File | Purpose |
|---|---|
| `ads/state.py` | The shared contracts: `Evidence`, `Claim`, `Feedback`, `ToolResult`, `AgentState`. **Keep stable.** |
| `ads/ledger.py` | Append-only evidence store: IDs, caching, save/load, `digest()` text for LLMs. |
| `ads/context.py` | `RunContext`: ledger + current dataframe + dataset version + `runs/<time>/` folder + in-memory `cache`. |
| `ads/verify.py` | Claim verifier (numbers must match cited evidence). Heuristics are documented at the top of the file. |
| `ads/config.py` | All settings; loads `.env`; picks the LLM provider. |
| `ads/tools/registry.py` | `@register` decorator and `execute_tool()`, the **only** way tools run (writes the ledger, handles caching and data versions). |
| `ads/tools/data.py` | `profile_dataset`, `clean_data`, `identify_target` |
| `ads/tools/stats.py` | `test_association` (right test per type, Benjamini-Hochberg correction, effect sizes), `segment_rates`, `top_correlations` |
| `ads/tools/ml.py` | `train_cv` (CV on train split, test set touched once, baselines), `permutation_importance`, `detect_leakage` |
| `ads/tools/viz.py` | `plot_target_by_feature`, `plot_importance`, `plot_correlation` |
| `ads/llm/client.py` | LLM client: modes live / offline / fake; providers Gemini (free) and Anthropic; disk cache; throttling; retries |
| `ads/llm/gemini.py` | Translation between our message format and Gemini's (including thought signatures) |
| `ads/llm/loop.py` | `run_agent_loop()`: the generic tool-calling loop every agent will reuse |
| `tests/` | ~55 tests. Synthetic data in `tests/conftest.py` |
| `scripts/smoke_tools.py`, `scripts/check_llm.py` | Real-data and real-LLM sanity checks |

**Not verified yet:** the Gemini adapter against the live API (run `check_llm.py`), and anything in
`tests/test_gemini_adapter.py` that needs the `google-genai` package. If `check_llm.py` fails, fix that
first. Likely culprits: the model name (set `ADS_MODEL` to an exact id shown in AI Studio) or the schema
conversion in `ads/llm/gemini.py`.

The **old prototype is still in the repo root** (`agents/`, `graph.py`, `app.py`, `evaluation/`,
`tools/`). It is being replaced, not extended. Do not import from it. Move it to `legacy/` on the last day.

---

## 4. The architecture you are completing

```
            goal + CSV
                |
          Orchestrator  --plans-->  [data, eda, ml, viz, insight, critic]
                |
   Data -> EDA -> ML -> Viz -> Insight (writes Claims) -> Critic
                                                            |
                          blockers found?  yes -> Feedback to ONE target agent,
                                                  re-run from that agent onward
                                           no  -> Report
```

Each specialist agent is `run_agent_loop(...)` with its own system prompt and tool allow-list. If the LLM
is unavailable, the same agent runs a fixed **scripted policy** (same tools, fixed order). Both paths write
to the same ledger, so everything downstream is identical.

Ground rules:
1. Never let an LLM produce a number. Numbers come from tools only.
2. Every tool call goes through `execute_tool()`.
3. Agents return partial state updates; they do not mutate shared state in place.
4. If you change a shape in `state.py`, change every user of it in the same commit.
5. Tests stay green before every commit. Offline mode stays working.

---

## 5. What is left, with specs

### 5.1 `drop_columns` tool (small, do first)
Add to `ads/tools/data.py`: `drop_columns(df, columns)`, a transform (`cacheable=False`, returns
`dataframe=`), summary like "Dropped 1 column(s): ChurnFlag". This is how the Data agent acts on leakage
feedback. Test: version bumps, columns gone, ledger entry created.

### 5.2 Agent base + scripted policy (`ads/agents/base.py`, `ads/agents/policy.py`)
```python
def run_specialist(state, agent_name, system, task, tool_names, fallback) -> dict
```
If `client.available`: call `run_agent_loop`; on `LLMUnavailable` fall back. Otherwise call
`fallback(ctx, state)`. Add `client: Any` to `AgentState`. Return `{"trace": [...]}`.

Scripted policies (target and problem type are read from the latest `identify_target` evidence):

| Agent | Offline sequence |
|---|---|
| data | `profile_dataset` -> `identify_target(goal)` -> `clean_data(target)` -> `detect_leakage(target)` |
| eda | `test_association` -> `top_correlations` -> `segment_rates` for the 3 most significant features (binary or regression targets only) |
| ml | `train_cv(target, exclude=...)` -> `permutation_importance` |
| viz | `plot_importance`, `plot_target_by_feature` for the top 2 features, `plot_correlation` |

### 5.3 The four analysis agents (`ads/agents/{data,eda,ml,viz}.py`)
Thin wrappers: a system prompt (role, what to do, "use only tool results, never invent numbers, stop when
done"), a task string built from the goal and ledger digest, a tool allow-list, and the fallback. Each is
`def run(state) -> dict`.

### 5.4 Insight agent (`ads/agents/insight.py`)
Output: `List[Claim]`. Live mode: the LLM gets `ledger.digest()` and must return JSON claims
`{"text": ..., "evidence_ids": [...]}`. Offline mode: template claims built from evidence results (churn
rate by segment, strongest effects, model scores, top predictors). Then run `verify_claims`; failures become
`Feedback` (already implemented in `rejections_to_feedback`), and the agent gets up to
`SETTINGS.max_retries_per_agent` rewrite attempts. Claims still failing are excluded and listed in the
report's audit appendix. Wording rule: associations, never "causes" or promised percentage reductions.
Recommendations must be derived from segment and driver evidence.

### 5.5 Critic (`ads/agents/critic.py`)
Programmatic checks, each producing a `Feedback` with a `target_agent`:

| Check | Evidence used | Rule (starting point) | Severity | Routes to |
|---|---|---|---|---|
| Leakage | `detect_leakage` suspects | any suspect | blocker | data_agent (drop column) |
| Implausible score | `train_cv` test ROC-AUC | > 0.98 and no leakage check run | warning | data_agent |
| Tiny sample | `train_cv` n_rows_used | < 200 rows | warning | insight_agent (add caveat) |
| Overfitting | `train_vs_cv_gap` | > 0.10 | warning | ml_agent |
| Unstable CV | `cv_best_std` | std > 0.05 | warning | ml_agent |
| Imbalance | `minority_share`, baseline accuracy | minority < 20%: claims must cite ROC-AUC/F1, not accuracy alone | warning | insight_agent |
| Multiplicity | `test_association` | always note the correction was applied | info | none |
| Unsupported claims | verifier | any rejected claim | blocker | insight_agent |
| Causal language | claim text regex (`causes`, `will reduce`, `guarantee`) | any match | warning | insight_agent |

Confidence score: a **documented rubric**, not an arbitrary subtraction. Define a weighted checklist in
`config.py`; `confidence = 100 * passed_weight / total_weight`. Document it in `docs/evaluation.md`.
An optional LLM review layer (logical gaps) is a nice-to-have; cut it first if time is short.

### 5.6 Orchestrator and graph (`ads/agents/orchestrator.py`, `ads/graph.py`)
LangGraph `StateGraph` over `AgentState`. Nodes: orchestrator, data, eda, ml, viz, insight, critic,
finalize. Orchestrator: LLM proposes a plan with reasons (fallback: the fixed chain); after each stage a
short "reflect" step may add requirements (for example, imbalanced target -> report precision and recall).
Conditional edge after critic: if blockers exist and the retry budget allows, jump to the `target_agent` of
the first blocker and re-run from there. Budgets (`max_agent_steps`, `max_retries_per_agent`,
`max_replans`) are already in `config.py`. Because tools are cached per dataset version, re-running
unchanged stages is free.

Worked example of the loop: Critic finds `ChurnFlag` predicts churn alone (AUC 1.00) -> Feedback to
data_agent -> `drop_columns(["ChurnFlag"])` -> dataset becomes v2 -> eda/ml/viz/insight re-run on v2 ->
model AUC falls to a realistic 0.84 -> Critic passes.

### 5.7 Report renderer (`ads/report/render.py`)
Input: claims, chart paths, critic findings, confidence, ledger. Output: HTML and Markdown (PDF optional,
cut first). Each number links to its evidence ID. Include an **audit appendix**: the evidence table,
rejected claims, critic findings, and the confidence rubric result.

### 5.8 Entry points (`ads/pipeline.py`, `ads/cli.py`)
`run_analysis(dataset_path, goal, client=None) -> RunResult` builds the context, runs the graph, saves
`runs/<id>/{ledger.json, report.html, report.md, charts/}`. CLI: `python -m ads.cli data.csv "goal"`.
Integration test: run the full pipeline offline on `churn_df` and assert the report exists and all claims
verify.

### 5.9 Evaluation (`evaluation/`, the part that decides the viva)
**Cases:** each of the 4 datasets x seeded flaws, 8 cases minimum.

| Flaw | How to inject | Ground truth |
|---|---|---|
| none (control) | unchanged | no flag expected (measures false alarms) |
| leakage | add a column that is a copy of the target | flag naming that column |
| row-order leak | sort by target, add a running row number | flag naming that column |
| tiny sample | keep only 40 rows | small-sample flag |
| heavy missingness | blank 40% of cells in key features | cleaning noted, no crash |

**Systems:** B1 = single-prompt LLM (data summary in, report out, no tools); B2 = our pipeline without the
Critic; B3 = full pipeline.

**Metrics:** flaw detection rate; false-alarm rate on controls; **number grounding rate** (share of numbers
in the report that match a reference ledger from running the tools on the same data; B1 will score poorly);
score honesty (reported AUC vs the reference held-out AUC); completion rate; runtime and LLM calls. Run each
case 3 times if budget allows. Output tables into `docs/evaluation.md`.

Free-tier planning: B2 and B3 can run **offline** (flaw detection is programmatic), so only B1 truly needs
the LLM. Responses are cached on disk, so re-runs are free. Check your daily request limit in AI Studio
before a big run, and run the benchmark in one sitting.

### 5.10 App and polish
Rework `app.py` (Streamlit): upload + goal, live agent trace, evidence table with click-through from any
number, verified vs rejected claims panel, confidence with its rubric, charts, download buttons. Then: move
old files to `legacy/`, rewrite `README.md`, handle edge cases (no clear target, all-numeric data, tiny or
malformed CSV; fail with a clear message, never a stack trace), add `runs/` and `.llm_cache/` to
`.gitignore`.

### 5.11 Documents
`docs/architecture.md`, `docs/evaluation.md` (method, tables, honest limits), final report, slides, demo
script, and viva Q&A (see section 8).

---

## 6. Build order and time budget

Do the steps in this order; each ends with something that runs and passes tests.

| Step | Work | Rough time |
|---|---|---|
| 0 | Setup, `pytest`, `smoke_tools.py`, `check_llm.py` | 0.5 h |
| 1 | 5.1 `drop_columns`, then 5.2 base + policies, 5.3 agents | 3 h |
| 2 | 5.4 Insight agent + verifier loop | 2.5 h |
| 3 | 5.5 Critic | 3 h |
| 4 | 5.6 graph + 5.8 pipeline/CLI. **Checkpoint: one command, full offline run, report with evidence links** | 3 h |
| 5 | 5.7 report renderer | 2 h |
| 6 | 5.9 benchmark and results | 4 h |
| 7 | 5.10 app + cleanup + edge cases | 3 h |
| 8 | 5.11 documents, demo rehearsal on a clean machine | 4 h |

Total is about 25 hours. If time runs short, cut in this order: PDF export, the Critic's LLM review layer,
extra benchmark cases (never below 8). **Never cut** the ledger, the verifier, or the benchmark.

---

## 7. Known risks and gotchas

- **Library versions:** developed on Python 3.12, pandas 3.1, scikit-learn 1.9. Tests that train models
  take about 45 seconds in total.
- **Gemini free tier:** low requests-per-minute and a daily cap, and free-tier prompts may be used by Google
  to improve its products (our LLM only sees tool summaries and column names, never raw rows). The client
  throttles (`ADS_MIN_INTERVAL`, default 4 s for Gemini), retries with backoff, and caches. If the model name
  fails, set `ADS_MODEL` to an exact id from AI Studio.
- **Verifier limits (be honest about them in the report):** bare integers 0-10 are ignored; "p < 0.001"
  matches any cited number in that range. It proves numbers are traceable and correctly quoted. It does not
  prove the analysis was the right one or that an association is causal.
- **Cached LLM answers** live in `.llm_cache/`. Delete it when you change prompts, or you will keep getting
  old answers.
- Do not retrain models inside agents; use `train_cv`, which caches the fitted pipeline in `ctx.cache`.

---

## 8. Questions the examiners will ask (prepare answers)

1. What makes this agentic rather than a pipeline? (planner chooses tools, replans, critic-driven retries)
2. How do you stop the LLM hallucinating numbers? (tools compute, ledger records, verifier checks)
3. What does the Critic catch that a plain LLM misses? (show the leakage benchmark)
4. What are the limits of the verifier? (section 7)
5. Why Benjamini-Hochberg? (30 tests at 5% gives 1-2 false positives by chance)
6. Why a held-out test set and CV on train only? (model selection on test data inflates scores)
7. How did you evaluate, and what are the weaknesses of your evaluation?
8. What happens with no internet? (offline policy, same ledger and report)

---

## 9. Working agreements

Branch per task, small commits, `python -m pytest` green before merging. Update the checkboxes in
`docs/PLAN.md` as you finish items. When you finish a step, run `python scripts/smoke_tools.py` once to
catch real-data breakage that synthetic tests cannot.
