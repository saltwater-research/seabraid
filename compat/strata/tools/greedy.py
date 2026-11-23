#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""greedy.py URL OUT [--model M] [--max-tokens N] [--timeout S] [--prompt TEXT]

Send one greedy (temperature 0, thinking disabled) chat-completion request to an
OpenAI-compatible /v1/chat/completions endpoint and write {"text", "tokens", "secs"} as JSON to OUT.
Arguments:
  URL            full chat-completions endpoint, e.g. http://127.0.0.1:8080/v1/chat/completions
  OUT            JSON result file to write
  --model        model field in the request body (default: x; single-model servers ignore it)
  --max-tokens   completion cap (default: 400)
  --timeout      request timeout in seconds (default: 600)
  --prompt       override the built-in MoE routing prompt
"""
import argparse
import json
import time

from _loopback import post_json

DEFAULT_PROMPT = ("Explain in detail how a mixture-of-experts transformer routes each token to its experts, "
                  "and why expert placement across memory tiers affects decode speed.")


def main():
    ap = argparse.ArgumentParser(description="One greedy chat completion; writes the text, token count and wall time.")
    ap.add_argument("url", help="chat-completions endpoint URL")
    ap.add_argument("out", help="JSON result file to write")
    ap.add_argument("--model", default="x", help="model field in the body (default: x)")
    ap.add_argument("--max-tokens", type=int, default=400, help="completion cap (default: 400)")
    ap.add_argument("--timeout", type=float, default=600, help="request timeout in seconds (default: 600)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt text (default: built-in MoE prompt)")
    a = ap.parse_args()

    body = {"model": a.model, "messages": [{"role": "user", "content": a.prompt}],
            "max_tokens": a.max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    d = post_json(a.url, body, a.timeout)
    txt = d["choices"][0]["message"].get("content") or ""
    n = d.get("usage", {}).get("completion_tokens")
    with open(a.out, "w") as fh:
        json.dump({"text": txt, "tokens": n, "secs": time.time() - t}, fh)
    print(n, "tokens", round(time.time() - t, 1), "s")


if __name__ == "__main__":
    main()
