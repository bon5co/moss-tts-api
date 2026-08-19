.PHONY: server worker fake-worker health smoke test

# --- the TTS server, on the machine with the models ---
server:             ## run the MOSS-TTS server
	uv run --extra server python -m app.main

# --- the Temporal worker; no models, runs anywhere ---
worker:             ## run the Temporal worker in the foreground
	uv run --extra worker moss-worker

fake-worker:        ## same workflow, silent audio, no TTS server needed
	uv run --extra worker python scripts/fake_worker.py

# --- anywhere that can reach Temporal ---
health:
	uv run --extra worker moss health

smoke:
	uv run --extra worker moss submit "Notice. The speech worker is running." --wait

test:
	uv run --extra server pytest -q
