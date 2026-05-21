"""Thin client for server.py. Sends one JSON command, streams output.

Stdlib-only; does NOT import isaaclab or torch — runs fast, no sim startup.

Ctrl+C closes the socket; the server detects the disconnect and stops
its current operation, then returns to listening for the next command.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

DEFAULT_SOCKET = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".server.sock")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send a command to the cartpole_ws server.")
    p.add_argument("--socket", default=DEFAULT_SOCKET, help="Path to server socket.")
    sub = p.add_subparsers(dest="action", required=True)

    pt = sub.add_parser("train", help="Train a fresh policy (or resume from checkpoint).")
    pt.add_argument("--max_iters", type=int, default=200)
    pt.add_argument("--rollout_steps", type=int, default=128)
    pt.add_argument("--run_name", type=str, default=None)
    pt.add_argument("--resume_from", type=str, default=None, help="Optional .pt to warm-start from.")
    pt.add_argument("--action_std_init", type=float, default=0.6)
    pt.add_argument("--action_std_decay_rate", type=float, default=0.05)
    pt.add_argument("--min_action_std", type=float, default=0.10)
    pt.add_argument("--action_std_decay_freq", type=int, default=20)
    pt.add_argument("--save_freq", type=int, default=25)

    pp = sub.add_parser("play", help="Run a saved policy.")
    pp.add_argument("--checkpoint", type=str, required=True)
    pp.add_argument("--play_steps", type=int, default=1500)
    pp.add_argument("--deterministic", action="store_true")
    pp.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse()
    sock_path = args.socket
    if not os.path.exists(sock_path):
        print(
            f"ERROR: server socket {sock_path} not found.\n"
            "Start the server first:  make env",
            file=sys.stderr,
        )
        return 1

    # Convert argparse Namespace -> dict for the wire protocol.
    payload = vars(args).copy()
    payload.pop("socket")

    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"ERROR: cannot connect to {sock_path}: {e}", file=sys.stderr)
        return 1

    s.sendall((json.dumps(payload) + "\n").encode())

    rc = 0
    try:
        f = s.makefile("r", encoding="utf-8")
        for line in f:
            line = line.rstrip("\n")
            if line == "DONE":
                break
            if line.startswith("ERROR"):
                rc = 1
            print(line, flush=True)
    except KeyboardInterrupt:
        print("\n[client] Ctrl+C — closing connection (server will abort current op).",
              flush=True)
    finally:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        s.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
