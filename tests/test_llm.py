import pytest

from ads.llm import LLMClient, LLMResponse, LLMUnavailable


def test_fake_mode_replays_the_script_in_order_then_raises():
    client = LLMClient(mode="fake", script=[LLMResponse.from_text("one"), LLMResponse.from_text("two")])
    assert client.available
    assert client.complete("s", []).text == "one"
    assert client.complete("s", []).text == "two"
    with pytest.raises(LLMUnavailable):
        client.complete("s", [])


def test_fake_script_items_can_be_callables_that_see_the_request():
    seen = []

    def respond(system, messages, tools):
        seen.append((system, len(messages)))
        return LLMResponse.from_text("ok")

    client = LLMClient(mode="fake", script=[respond])
    assert client.complete("sys prompt", [{"role": "user", "content": "hi"}]).text == "ok"
    assert seen == [("sys prompt", 1)]


def test_offline_mode_is_unavailable():
    client = LLMClient(mode="offline")
    assert not client.available
    with pytest.raises(LLMUnavailable):
        client.complete("s", [])


def test_live_mode_answers_identical_requests_from_the_cache(tmp_path, monkeypatch):
    client = LLMClient(mode="live", cache_dir=tmp_path)
    api_calls = []

    def fake_api(system, messages, tools, max_tokens):
        api_calls.append(1)
        return LLMResponse.from_text("hello")

    monkeypatch.setattr(client, "_call_api", fake_api)
    msgs = [{"role": "user", "content": "hi"}]
    assert client.complete("sys", msgs).text == "hello"
    assert client.complete("sys", msgs).text == "hello"
    assert len(api_calls) == 1 and client.cache_hits == 1

    client.complete("sys", [{"role": "user", "content": "something else"}])
    assert len(api_calls) == 2


def test_cache_survives_a_new_client_instance(tmp_path, monkeypatch):
    first = LLMClient(mode="live", cache_dir=tmp_path)
    monkeypatch.setattr(first, "_call_api", lambda *a: LLMResponse.from_text("remembered"))
    first.complete("sys", [{"role": "user", "content": "q"}])

    second = LLMClient(mode="live", cache_dir=tmp_path)

    def must_not_be_called(*a):
        raise AssertionError("the API should not be called on a cache hit")

    monkeypatch.setattr(second, "_call_api", must_not_be_called)
    assert second.complete("sys", [{"role": "user", "content": "q"}]).text == "remembered"


def test_tool_call_helper_builds_anthropic_style_blocks():
    resp = LLMResponse.from_tool_calls(("profile_dataset", {"x": 1}), text="checking")
    assert resp.text == "checking"
    call = resp.tool_calls[0]
    assert call["name"] == "profile_dataset" and call["input"] == {"x": 1} and call["id"]
