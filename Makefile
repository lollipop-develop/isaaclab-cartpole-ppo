# Makefile for cartpole_ws PPO + Isaac Lab experiments.
#
# Workflow:
#   1. `make env`            in terminal A — starts Isaac Sim + server (persistent)
#   2. `make play`/`make train` in terminal B — short-lived clients
#   3. Ctrl+C in terminal B  — kills the client; server stays alive
#   4. Ctrl+C in terminal A  — shuts the server down
#
# Override variables on the command line, e.g.:
#   make env NUM_ENVS=512
#   make train MAX_ITERS=400 RUN_NAME=swingup_1
#   make play CHECKPOINT=runs/swingup_1/policy_final.pt PLAY_DET=1

SHELL := /bin/bash

# --- paths / env -------------------------------------------------------------
ISAACLAB       ?= $(HOME)/IsaacLab/isaaclab.sh
CONDA_SH       ?= $(HOME)/miniconda3/etc/profile.d/conda.sh
CONDA_ENV      ?= isaaclab
ACTIVATE       := source $(CONDA_SH) && conda activate $(CONDA_ENV)
SOCKET         ?= .server.sock

# --- server / training defaults ----------------------------------------------
# Environment to load: cartpole | double
ENV            ?= cartpole
NUM_ENVS       ?= 256
MAX_ITERS      ?= 200
ROLLOUT_STEPS  ?= 256
SEED           ?= 42
RUN_NAME       ?=
RESUME         ?=

# --- play defaults -----------------------------------------------------------
CHECKPOINT     ?=
PLAY_STEPS     ?= 1500
PLAY_DET       ?=                # set to 1 for deterministic playback

# --- GUI toggle (env target default = GUI; HEADLESS=1 to disable) ------------
HEADLESS_FLAG  := $(if $(HEADLESS),--headless,)

.DEFAULT_GOAL := help
.PHONY: help env env-headless train play tensorboard \
        smoke train-once play-once clean clean-runs clean-pyc kill-server

help:
	@echo "Persistent server workflow:"
	@echo "  make env              - boot Isaac Sim + server (foreground, GUI on)"
	@echo "  make env ENV=double   - boot the server with the double-pendulum env"
	@echo "  make env HEADLESS=1   - boot server headless"
	@echo "  make train            - send a train command to running server"
	@echo "  make play             - send a play command to running server"
	@echo "  make kill-server      - pkill the server (use if Ctrl+C in env terminal didn't work)"
	@echo ""
	@echo "Standalone (no server, each call boots Isaac Sim from scratch):"
	@echo "  make smoke            - 5-iter smoke test"
	@echo "  make train-once       - one-shot training"
	@echo "  make play-once        - one-shot playback"
	@echo ""
	@echo "Utility:"
	@echo "  make tensorboard      - tensorboard on runs/"
	@echo "  make clean | clean-runs | clean-pyc"
	@echo ""
	@echo "Vars: ENV NUM_ENVS MAX_ITERS ROLLOUT_STEPS SEED RUN_NAME RESUME"
	@echo "      CHECKPOINT PLAY_STEPS PLAY_DET HEADLESS"
	@echo "      ISAACLAB CONDA_ENV SOCKET"

# =====================================================================
# Persistent server (recommended workflow)
# =====================================================================

env:
	@if [ -S $(SOCKET) ]; then \
	    echo "[make] WARNING: $(SOCKET) already exists — old server still running?"; \
	    echo "       run 'make kill-server' first if needed."; \
	fi
	$(ACTIVATE) && $(ISAACLAB) -p server.py $(HEADLESS_FLAG) \
	    --socket $(SOCKET) --num_envs $(NUM_ENVS) --seed $(SEED) --env $(ENV)

env-headless:
	$(MAKE) env HEADLESS=1

train:
	@if [ ! -S $(SOCKET) ]; then \
	    echo "ERROR: $(SOCKET) not found. Start the server first: make env"; \
	    exit 1; \
	fi
	$(ACTIVATE) && python client.py --socket $(SOCKET) train \
	    --max_iters $(MAX_ITERS) --rollout_steps $(ROLLOUT_STEPS) \
	    $(if $(RUN_NAME),--run_name $(RUN_NAME),) \
	    $(if $(RESUME),--resume_from $(RESUME),)

play:
	@if [ ! -S $(SOCKET) ]; then \
	    echo "ERROR: $(SOCKET) not found. Start the server first: make env"; \
	    exit 1; \
	fi
	@CKPT="$(CHECKPOINT)"; \
	if [ -z "$$CKPT" ]; then \
	    CKPT=$$(ls -t runs/*/policy_final.pt 2>/dev/null | head -1); \
	fi; \
	if [ -z "$$CKPT" ]; then \
	    echo "ERROR: no checkpoint in runs/. Set CHECKPOINT=path/to/policy.pt"; \
	    exit 1; \
	fi; \
	echo "[make] using checkpoint: $$CKPT"; \
	$(ACTIVATE) && python client.py --socket $(SOCKET) play \
	    --checkpoint "$$CKPT" --play_steps $(PLAY_STEPS) \
	    $(if $(PLAY_DET),--deterministic,)

kill-server:
	@# Isaac Sim absorbs SIGTERM/SIGINT in some code paths — use SIGKILL.
	@# The .* matches whether or not --headless sits between server.py and
	@# --socket; this single pattern covers both the python process and the
	@# isaaclab.sh wrapper (both command lines contain server.py ... --socket).
	-pkill -9 -f "server.py.*--socket $(SOCKET)" || true
	-rm -f $(SOCKET)
	@echo "[make] killed server processes and removed $(SOCKET)"

# =====================================================================
# Standalone scripts (boot Isaac Sim from scratch each call)
# =====================================================================

smoke:
	$(ACTIVATE) && $(ISAACLAB) -p train.py --headless \
	    --num_envs 64 --max_iters 5 --rollout_steps 64 --env $(ENV)

train-once:
	$(ACTIVATE) && $(ISAACLAB) -p train.py $(HEADLESS_FLAG) \
	    --num_envs $(NUM_ENVS) --max_iters $(MAX_ITERS) \
	    --rollout_steps $(ROLLOUT_STEPS) --seed $(SEED) --env $(ENV) \
	    $(if $(RUN_NAME),--run_name $(RUN_NAME),)

play-once:
	@CKPT="$(CHECKPOINT)"; \
	if [ -z "$$CKPT" ]; then \
	    CKPT=$$(ls -t runs/*/policy_final.pt 2>/dev/null | head -1); \
	fi; \
	if [ -z "$$CKPT" ]; then \
	    echo "ERROR: no checkpoint in runs/. Set CHECKPOINT=path/to/policy.pt"; \
	    exit 1; \
	fi; \
	echo "[make] using checkpoint: $$CKPT"; \
	$(ACTIVATE) && $(ISAACLAB) -p play.py $(HEADLESS_FLAG) \
	    --checkpoint "$$CKPT" --num_envs $(NUM_ENVS) --num_steps $(PLAY_STEPS) \
	    --env $(ENV) $(if $(PLAY_DET),--deterministic,)

# =====================================================================
# Utility
# =====================================================================

tensorboard:
	$(ACTIVATE) && tensorboard --logdir runs/

clean: clean-runs clean-pyc

clean-runs:
	rm -rf runs/

clean-pyc:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
