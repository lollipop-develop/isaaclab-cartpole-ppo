"""Long-running Isaac Lab + PPO server.

Boots Isaac Sim once and listens on a Unix-domain socket. Each connection
sends a JSON command (train/play); output lines are streamed back and the
connection closes when the operation completes or the client disconnects.

When a client disconnects mid-operation, the server aborts that operation
and returns to listening, ready for the next command. Ctrl+C in this
server's terminal cleanly shuts the whole thing down.

Edit ``cartpole_env.py`` to change state / reward — the server picks up
the new code at next startup.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback

from isaaclab.app import AppLauncher

# ---- CLI ----
_here = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Persistent Isaac Lab + PPO server.")
parser.add_argument("--socket", default=os.path.join(_here, ".server.sock"))
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--env", default="cartpole", choices=["cartpole", "double"],
                    help="Which environment to load for this server session.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Now safe to import heavy deps ----
import torch  # noqa: E402

from env_registry import ENVS  # noqa: E402

torch.manual_seed(args_cli.seed)

# ---- Build env once (each env carries its own PPO module) ----
_env_cls, _cfg_cls, _ppo_mod = ENVS[args_cli.env]
cfg = _cfg_cls()
cfg.scene.num_envs = args_cli.num_envs
env = _env_cls(cfg=cfg, render_mode=None)
device = env.device
print(
    f"[server] env={args_cli.env} ready  num_envs={args_cli.num_envs}  device={device}  "
    f"obs={cfg.observation_space}  act={cfg.action_space}  "
    f"episode_s={cfg.episode_length_s}",
    flush=True,
)


# =====================================================================
# Helpers
# =====================================================================
def send(conn: socket.socket, msg: str) -> bool:
    """Send one line. Returns False if client is gone."""
    try:
        conn.sendall((msg + "\n").encode())
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def client_alive(conn: socket.socket) -> bool:
    """Non-blocking check that the client socket is still connected."""
    try:
        conn.setblocking(False)
        try:
            data = conn.recv(1, socket.MSG_PEEK)
            if data == b"":
                return False
        except BlockingIOError:
            return True
        except (ConnectionResetError, OSError):
            return False
        finally:
            conn.setblocking(True)
        return True
    except OSError:
        return False


def make_ppo(action_std_init: float):
    """Build this env's PPO agent. lr/gamma/K/eps come from the env's HYPERPARAMS."""
    hp = _ppo_mod.HYPERPARAMS
    return _ppo_mod.PPO(
        state_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        lr_actor=hp["lr_actor"], lr_critic=hp["lr_critic"],
        gamma=hp["gamma"], K_epochs=hp["K_epochs"], eps_clip=hp["eps_clip"],
        action_std_init=action_std_init, device=device,
        gae_lambda=hp["gae_lambda"],
    )


# =====================================================================
# Command handlers
# =====================================================================
def cmd_train(conn: socket.socket, args: dict) -> None:
    # Defaults for env hyperparameters come from the env's HYPERPARAMS dict;
    # a client may still override any of them per-command.
    hp = _ppo_mod.HYPERPARAMS
    max_iters = int(args.get("max_iters", 200))
    rollout_steps = int(args.get("rollout_steps", 128))
    action_std_init = float(args.get("action_std_init", hp["action_std_init"]))
    action_std_decay_rate = float(args.get("action_std_decay_rate", hp["action_std_decay_rate"]))
    min_action_std = float(args.get("min_action_std", hp["min_action_std"]))
    action_std_decay_freq = int(args.get("action_std_decay_freq", hp["action_std_decay_freq"]))
    save_freq = int(args.get("save_freq", hp["save_freq"]))
    resume_from = args.get("resume_from")
    run_name = args.get("run_name") or time.strftime("%Y%m%d-%H%M%S")

    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    ppo = make_ppo(action_std_init)
    if resume_from:
        if not os.path.exists(resume_from):
            send(conn, f"ERROR: resume_from not found: {resume_from}")
            return
        ppo.load(resume_from)
        send(conn, f"[server] resumed weights from {resume_from}")

    send(conn, f"[server] training: max_iters={max_iters} rollout={rollout_steps} run_name={run_name}")

    obs_dict, _ = env.reset()
    state = obs_dict["policy"]
    ep_r = torch.zeros(args_cli.num_envs, device=device)
    ep_l = torch.zeros(args_cli.num_envs, device=device)
    t0 = time.time()
    total_steps = 0

    for it in range(max_iters):
        ep_returns: list[float] = []
        ep_lengths: list[float] = []

        for _ in range(rollout_steps):
            action = ppo.select_action(state)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)

            ep_r += reward
            ep_l += 1
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                ep_returns.extend(ep_r[idx].cpu().tolist())
                ep_lengths.extend(ep_l[idx].cpu().tolist())
                ep_r[idx] = 0.0
                ep_l[idx] = 0.0
            state = obs_dict["policy"]
            total_steps += args_cli.num_envs

        # `state` here is the env's obs AFTER the last buffered step — used by
        # GAE in ppo_double.PPO.update() to bootstrap V(s_T); ignored by ppo_cartpole.
        ppo.update(next_state=state)

        if (it + 1) % action_std_decay_freq == 0:
            ppo.decay_action_std(action_std_decay_rate, min_action_std)

        mean_ret = sum(ep_returns) / max(len(ep_returns), 1)
        mean_len = sum(ep_lengths) / max(len(ep_lengths), 1)
        sps = total_steps / (time.time() - t0 + 1e-9)
        if not send(
            conn,
            f"iter {it+1:4d}/{max_iters}  "
            f"ep_ret={mean_ret:7.2f}  ep_len={mean_len:6.1f}  "
            f"n_eps={len(ep_returns):4d}  std={ppo.action_std:.3f}  "
            f"sps={sps:7.0f}",
        ):
            print(f"[server] client gone at iter {it+1}, aborting train", flush=True)
            return

        if (it + 1) % save_freq == 0:
            ckpt = os.path.join(run_dir, f"policy_{it+1:04d}.pt")
            ppo.save(ckpt)
            send(conn, f"[server] saved {ckpt}")

        if not client_alive(conn):
            print(f"[server] client disconnected at iter {it+1}, aborting train", flush=True)
            return

    final = os.path.join(run_dir, "policy_final.pt")
    ppo.save(final)
    send(conn, f"[server] saved {final}")


def cmd_play(conn: socket.socket, args: dict) -> None:
    checkpoint = args.get("checkpoint", "")
    play_steps = int(args.get("play_steps", 1500))
    deterministic = bool(args.get("deterministic", False))
    log_every = int(args.get("log_every", 50))

    if not checkpoint:
        send(conn, "ERROR: no checkpoint specified")
        return
    if not os.path.exists(checkpoint):
        send(conn, f"ERROR: checkpoint not found: {checkpoint}")
        return

    ppo = make_ppo(action_std_init=0.10)
    ppo.load(checkpoint)
    send(conn, f"[server] loaded {checkpoint} (deterministic={deterministic})")

    obs_dict, _ = env.reset()
    state = obs_dict["policy"]
    ep_returns: list[float] = []
    ep_lengths: list[float] = []
    ep_r = torch.zeros(args_cli.num_envs, device=device)
    ep_l = torch.zeros(args_cli.num_envs, device=device)

    with torch.no_grad():
        for step in range(play_steps):
            if deterministic:
                action = ppo.policy_old.actor(state)
            else:
                action, _, _ = ppo.policy_old.act(state)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            ep_r += reward
            ep_l += 1
            if done.any():
                idx = done.nonzero(as_tuple=True)[0]
                ep_returns.extend(ep_r[idx].cpu().tolist())
                ep_lengths.extend(ep_l[idx].cpu().tolist())
                ep_r[idx] = 0.0
                ep_l[idx] = 0.0
            state = obs_dict["policy"]

            if step > 0 and step % log_every == 0:
                msg = f"step {step:5d}/{play_steps}  eps_done={len(ep_returns):4d}"
                if ep_returns:
                    msg += f"  last_ep_ret={ep_returns[-1]:7.2f}"
                if not send(conn, msg):
                    print(f"[server] client gone at step {step}, aborting play", flush=True)
                    return
                if not client_alive(conn):
                    print(f"[server] client disconnected at step {step}, aborting play", flush=True)
                    return

    if ep_returns:
        send(conn,
             f"[play] {len(ep_returns)} eps  "
             f"ep_return={sum(ep_returns)/len(ep_returns):.2f}  "
             f"ep_length={sum(ep_lengths)/len(ep_lengths):.1f}")
    else:
        send(conn, "[play] no eps completed in the given step budget")


HANDLERS = {"train": cmd_train, "play": cmd_play}


# =====================================================================
# Main loop
# =====================================================================
def _hard_exit(signum, frame):
    """Force exit on SIGINT/SIGTERM — Isaac Sim sometimes blocks normal shutdown."""
    print(f"\n[server] caught signal {signum}, hard-exiting.", flush=True)
    try:
        os.unlink(args_cli.socket)
    except FileNotFoundError:
        pass
    os._exit(0)


def main() -> None:
    sock_path = args_cli.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    os.chmod(sock_path, 0o600)
    # Install hard-exit handlers so Ctrl+C / SIGTERM in this terminal actually stops us.
    signal.signal(signal.SIGINT, _hard_exit)
    signal.signal(signal.SIGTERM, _hard_exit)
    print(f"[server] listening on {sock_path}  (Ctrl+C here to shut down)", flush=True)

    try:
        while True:
            conn, _ = srv.accept()
            try:
                # Read one line of JSON.
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(8192)
                    if not chunk:
                        break
                    data += chunk
                if not data.strip():
                    continue
                req = json.loads(data.decode().strip())
                action = req.pop("action", None)
                print(f"[server] cmd: {action} {req}", flush=True)
                handler = HANDLERS.get(action)
                if handler is None:
                    send(conn, f"ERROR: unknown action: {action}")
                else:
                    handler(conn, req)
                send(conn, "DONE")
            except (json.JSONDecodeError, KeyError) as e:
                send(conn, f"ERROR: bad request: {e}")
            except Exception:
                tb = traceback.format_exc()
                print(f"[server] error:\n{tb}", flush=True)
                send(conn, "ERROR:\n" + tb)
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
    except KeyboardInterrupt:
        print("\n[server] shutting down (Ctrl+C)", flush=True)
    finally:
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
