import json

import ads.tools  # noqa: F401
from ads.llm import LLMClient, LLMResponse, run_agent_loop


def test_loop_runs_tools_through_the_ledger_and_shows_evidence_ids_to_the_llm(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    seen = []

    def final(system, messages, tools):
        seen.append(json.dumps(messages[-1], default=str))
        return LLMResponse.from_text("Data looks clean.")

    client = LLMClient(
        mode="fake",
        script=[
            LLMResponse.from_tool_calls(("profile_dataset", {})),
            LLMResponse.from_tool_calls(("identify_target", {"goal": "why do customers churn"})),
            final,
        ],
    )
    result = run_agent_loop(client, ctx, "data_agent", "You are a data agent.", "Inspect the data.")

    assert result.stopped_reason == "done" and result.steps == 3
    assert result.final_text == "Data looks clean."
    assert [e.tool for e in ctx.ledger] == ["profile_dataset", "identify_target"]
    assert all(e.created_by == "data_agent" for e in ctx.ledger)
    assert "E0002" in seen[0]                       # the LLM was told which evidence it just created


def test_loop_stops_at_the_step_limit(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    forever = lambda system, messages, tools: LLMResponse.from_tool_calls(("profile_dataset", {}))
    client = LLMClient(mode="fake", script=[forever] * 5)
    result = run_agent_loop(client, ctx, "data_agent", "s", "t", max_steps=3)
    assert result.stopped_reason == "max_steps" and result.steps == 3
    assert len(ctx.ledger) == 1                     # identical repeated calls come from the cache


def test_tools_outside_the_allow_list_are_refused_and_hidden(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    seen, offered = [], []

    def final(system, messages, tools):
        seen.append(json.dumps(messages[-1], default=str))
        offered.append([t["name"] for t in tools])
        return LLMResponse.from_text("ok")

    client = LLMClient(mode="fake", script=[LLMResponse.from_tool_calls(("clean_data", {})), final])
    run_agent_loop(client, ctx, "viz_agent", "s", "t", tool_names=["plot_correlation"])

    assert len(ctx.ledger) == 0 and ctx.dataset_version == "v0"   # clean_data never ran
    assert "not available" in seen[0]
    assert offered == [["plot_correlation"]]


def test_a_failing_tool_is_reported_back_to_the_llm_not_raised(churn_df, make_ctx):
    ctx = make_ctx(churn_df)
    seen = []

    def final(system, messages, tools):
        seen.append(json.dumps(messages[-1], default=str))
        return LLMResponse.from_text("recovered")

    client = LLMClient(
        mode="fake",
        script=[LLMResponse.from_tool_calls(("identify_target", {"wrong_argument": 1})), final],
    )
    result = run_agent_loop(client, ctx, "data_agent", "s", "t")
    assert result.final_text == "recovered"
    assert "ERROR" in seen[0]
    assert result.trace[0]["ok"] is False and len(ctx.ledger) == 0
