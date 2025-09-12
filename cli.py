#!/usr/bin/env python
"""
Lightweight CLI for quick testing from the console.

Usage examples (PowerShell):
  # Streamed chat via running server (http://localhost:5000)
  python cli.py -m "Hello" --stream

  # Direct (no server) using local models only
  python cli.py -m "Explain transformers" --local

  # With web/RAG toggles
  python cli.py -m "Summarize https://example.com" --stream --web

  # Translate text
  python cli.py --translate en --text "Hola, que tal?"

  # Force GPU from CLI (Dev mode must be enabled in app for the endpoint)
  python cli.py --force-gpu --quantization 8bit

Notes:
  - By default talks to the server at http://localhost:5000.
  - If the server is unreachable and --local is not set, the CLI will fall back to direct local call.
"""

import argparse
import json
import sys
import time

DEFAULT_BASE = "http://localhost:5000"


def http_json(method, url, json_body=None, timeout=30):
    import requests
    resp = requests.request(method, url, json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def stream_sse(url, payload):
    import requests
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_lines(decode_unicode=True):
            if not chunk:
                continue
            if chunk.startswith("data: "):
                try:
                    obj = json.loads(chunk[6:])
                except Exception:
                    continue
                if "delta" in obj and obj["delta"]:
                    sys.stdout.write(obj["delta"])  # write partial tokens
                    sys.stdout.flush()
                if obj.get("done"):
                    break


def local_generate(message, web=False, rag=False, user=None):
    try:
        import chat as chat_mod
        res = chat_mod.get_response(message, username=user or "cli", use_web=web, use_rag=rag)
        if isinstance(res, dict):
            return res.get("content") or str(res)
        return str(res)
    except Exception as e:
        return f"[local error] {e}"


def main():
    ap = argparse.ArgumentParser(description="DIZI AI CLI")
    ap.add_argument('-m', '--message', help='Message to send')
    ap.add_argument('-u', '--user', default='cli', help='Username')
    ap.add_argument('--web', action='store_true', help='Enable web summarize')
    ap.add_argument('--rag', action='store_true', help='Enable vector RAG')
    ap.add_argument('--base', default=DEFAULT_BASE, help='Server base URL')
    ap.add_argument('--stream', action='store_true', help='Stream tokens')
    ap.add_argument('--local', action='store_true', help='Bypass HTTP and call local directly')
    # translate mode
    ap.add_argument('--translate', metavar='LANG', help='Translate input text to target language code')
    ap.add_argument('--text', help='Text for translation (use with --translate)')
    # device / force-gpu helpers
    ap.add_argument('--device', choices=['auto','cuda','cpu'], help='Set device via API (dev mode)')
    ap.add_argument('--force-gpu', action='store_true', help='Call /api/force-gpu (dev mode)')
    ap.add_argument('--quantization', choices=['8bit','4bit'], help='Quantization hint for force-gpu')
    args = ap.parse_args()

    # Translate path
    if args.translate:
        text = args.text or args.message or ''
        if not text:
            print('Provide --text or -m with --translate', file=sys.stderr)
            sys.exit(2)
        try:
            data = http_json('POST', f"{args.base}/api/translate", {"text": text, "target": args.translate})
            print(f"[{data.get('source','?')} -> {data.get('target','?')}]\n{data.get('translated','')}" )
        except Exception as e:
            print(f"[http error] {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Device / force-gpu helpers
    if args.device:
        try:
            data = http_json('POST', f"{args.base}/api/set-device", {"device": args.device})
            print(f"Device set: {data}")
        except Exception as e:
            print(f"[http error] {e}", file=sys.stderr)
        # continue; may also send a message in the same run

    if args.force_gpu:
        try:
            body = {"quantization": args.quantization} if args.quantization else {}
            data = http_json('POST', f"{args.base}/api/force-gpu", body)
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[http error] {e}", file=sys.stderr)
        # continue; may also send a message

    if not args.message:
        ap.print_help()
        return

    # Chat
    payload = {"user": args.user, "message": args.message, "web": args.web, "rag": args.rag}
    if args.local:
        print(local_generate(args.message, web=args.web, rag=args.rag, user=args.user))
        return
    # Try HTTP; fall back to local if server unreachable
    try:
        if args.stream:
            stream_sse(f"{args.base}/chat/stream", payload)
            print()
        else:
            data = http_json('POST', f"{args.base}/chat", payload)
            res = data.get('response')
            if isinstance(res, dict):
                print(res.get('content',''))
            else:
                print(str(res))
    except Exception as e:
        print(f"[http error] {e}; falling back to local\n", file=sys.stderr)
        print(local_generate(args.message, web=args.web, rag=args.rag, user=args.user))


if __name__ == '__main__':
    main()

