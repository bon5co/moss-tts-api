.PHONY: worker fake-worker health smoke

# --- on a machine that can reach the TTS server ---
worker:             ## run the Temporal worker in the foreground
	uv run moss-worker

fake-worker:        ## same workflow, silent audio, no TTS server needed
	uv run python scripts/fake_worker.py

# --- anywhere that can reach Temporal ---
health:
	uv run moss health

smoke:
	uv run moss submit "Notice. The worker is up." --wait
