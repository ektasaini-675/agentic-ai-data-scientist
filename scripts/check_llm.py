"""Check that your LLM key works and that tool-calling works end to end.

Usage (from the project folder):
    python scripts/check_llm.py

It makes two small live calls (the second one runs a real agent loop with the
profile_dataset tool). With a free Gemini key this costs nothing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ads.tools  # noqa: E402,F401
from ads.config import SETTINGS  # noqa: E402
from ads.context import RunContext  # noqa: E402
from ads.llm import LLMClient, run_agent_loop  # noqa: E402


def main() -> int:
    client = LLMClient()
    print(f"mode: {client.mode} | provider: {client.provider if client.mode == 'live' else '-'} | model: {client.model}")
    if not client.available:
        print("No API key found. Put GEMINI_API_KEY=... in the .env file (see .env.example), then run this again.")
        return 1

    if client.provider == "gemini":
        import os

        source = "GEMINI_API_KEY" if os.environ.get("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        key = os.environ.get(source, "")
        masked = f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "(too short)"
        print(f"key source: {source} | looks like: {masked} | length: {len(key)} (Gemini keys usually start 'AIza' and are 39 characters)")
        if "paste" in key.lower() or "your-key" in key.lower():
            print("The key is still the placeholder text. Open .env and paste your real key.")
            return 1
    try:
        reply = client.complete("Reply with exactly one word.", [{"role": "user", "content": "Say hello."}])
        print(f"1) plain reply: {reply.text!r}")
    except Exception as exc:
        print(f"1) FAILED: {type(exc).__name__}: {exc}")
        if "API_KEY_INVALID" in str(exc):
            print("   Google rejected the key. Check the 'key source' and 'length' line above:")
            print("   a wrong length, quotes/spaces around the key, or an old Windows environment variable can cause this.")
        else:
            print("   If this is a 'model not found' error, open Google AI Studio, copy an exact model id,")
            print("   and set ADS_MODEL=<that id> in your .env file.")
        return 1

    csv = ROOT / "datasets" / "raw" / "sample.csv"
    if not csv.exists():
        csv = next((ROOT / "datasets" / "raw").glob("*.csv"))
    ctx = RunContext.create(str(csv), goal="smoke check", base_dir=SETTINGS.runs_dir / "smoke")
    result = run_agent_loop(
        client,
        ctx,
        "data_agent",
        system="You are a data inspection agent. Use the tools provided; do not guess numbers.",
        task="Profile the dataset with the tool, then tell me how many rows and columns it has and cite the evidence id.",
        tool_names=["profile_dataset"],
        max_steps=4,
    )
    print(f"2) agent loop: {result.stopped_reason} after {result.steps} step(s)")
    print(f"   tools called: {[t['tool'] for t in result.trace]}")
    print(f"   ledger:\n{ctx.ledger.digest()}")
    print(f"   final answer: {result.final_text}")
    ok = result.stopped_reason == "done" and len(ctx.ledger) >= 1
    print("\nTool calling works end to end." if ok else "\nThe loop did not complete; send me this output.")
    print(f"tokens used: {client.usage}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
