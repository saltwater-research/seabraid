#!/usr/bin/env python3
# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""send.py URL [--text FILE] [--model M] [--timeout S]

Send one teacher-forced prompt (max_tokens 1, greedy, thinking disabled) to a chat-completions endpoint.
The prompt is the text of FILE; the server re-tokenizes it inside the chat template, identically in every arm.
Prints "ok <secs> s" on success.
Arguments:
  URL          chat-completions endpoint URL
  --text       prompt file (default: text.txt in the same directory as this script)
  --model      model field in the body (default: x)
  --timeout    request timeout in seconds (default: 900)
"""
import argparse
import os
import time

from _loopback import post_json

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="Send one teacher-forced prompt and report the wall time.")
    ap.add_argument("url", help="chat-completions endpoint URL")
    ap.add_argument("--text", default=os.path.join(HERE, "text.txt"), help="prompt file (default: text.txt beside this script)")
    ap.add_argument("--model", default="x", help="model field in the body (default: x)")
    ap.add_argument("--timeout", type=float, default=900, help="request timeout in seconds (default: 900)")
    a = ap.parse_args()

    with open(a.text) as fh:
        txt = fh.read()
    body = {"model": a.model, "messages": [{"role": "user", "content": txt}], "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    post_json(a.url, body, a.timeout)
    print("ok", round(time.time() - t, 1), "s")


if __name__ == "__main__":
    main()
