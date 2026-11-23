#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""conc.py URL N MAXTOK [--model M] [--timeout S]

Fires N concurrent greedy (temperature 0, thinking disabled) chat requests with distinct prompts and prints
the aggregate tokens/s and each request's tokens/s.
Arguments:
  URL          chat-completions endpoint URL
  N            number of concurrent requests
  MAXTOK       max_tokens per request
  --model      model field in the body (default: x)
  --timeout    per-request timeout in seconds (default: 1800)
No environment variables.
"""
import argparse
import threading
import time

from _loopback import post_json

TOPICS = ["mixture-of-experts routing", "GPU memory hierarchies", "speculative decoding", "KV cache compression",
          "PCIe vs NVLink bandwidth", "quantization of LLM weights", "expert parallelism", "page cache behaviour"]


def main():
    ap = argparse.ArgumentParser(description="N concurrent greedy requests; prints aggregate and per-request tok/s.")
    ap.add_argument("url", help="chat-completions endpoint URL")
    ap.add_argument("n", type=int, help="number of concurrent requests")
    ap.add_argument("maxtok", type=int, help="max_tokens per request")
    ap.add_argument("--model", default="x", help="model field in the body (default: x)")
    ap.add_argument("--timeout", type=float, default=1800, help="per-request timeout in seconds (default: 1800)")
    a = ap.parse_args()

    res = [None] * a.n

    def go(i):
        body = {"model": a.model, "messages": [{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} in depth, with examples."}],
                "max_tokens": a.maxtok, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        d = post_json(a.url, body, a.timeout)
        res[i] = (d.get("usage", {}).get("completion_tokens", 0), time.time() - t)

    t0 = time.time()
    th = [threading.Thread(target=go, args=(i,)) for i in range(a.n)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.time() - t0
    tot = sum(r[0] for r in res)
    print(f"N={a.n}: {tot} tokens in {wall:.1f} s = aggregate {tot/wall:.1f} tok/s; per request " + " ".join(f"{r[0]/r[1]:.1f}" for r in res))


if __name__ == "__main__":
    main()
