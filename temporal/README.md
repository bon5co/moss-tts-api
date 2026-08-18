# moss-temporal

Speech synthesis on the [MOSS-TTS server in this repo](../README.md) as a
Temporal workflow. An agent submits a script, polls for status, and gets URLs.

This is an **addition**, not a replacement. The HTTP API is unchanged and still
the right tool for a single short line you are going to block on. Worker mode
exists for the other case: a batch that takes ten minutes, submitted by
something that must not sit in a socket for ten minutes, and that should still
produce audio if the submitting process dies halfway.

## Why a queue at all

The measured numbers are the argument. The same 33-character line, synthesized
five times back to back on the same machine, took **14.9s / 17.7s / 34s / 54s /
78s** — a 5x spread, up to roughly 35x wall per second of audio. From idle the
server has usually dropped its weights (`IDLE_UNLOAD_SECONDS`, default 900) and
the first call pays a further **~127s** to load them.

A synchronous HTTP call across that distribution is a caller holding a socket
open for an unbounded time with no way to ask how far along it is, no retry
that does not redo the whole batch, and nothing written down if it dies. Those
are the four things Temporal already implements, so this does not implement
them again.

## Shape

```
agent ──start_workflow──►  Temporal (raphael host, :7233)
                             │  namespace moss, task queue moss-tts
                             ▼
                        worker (anywhere that can reach the TTS server)
                             │  one clip at a time
                             ├──POST /v1/audio/speech──► MOSS-TTS server (a Mac)
                             ├──PUT──► MinIO (raphael host, :9100, bucket moss)
                             └──result: audio URLs, not audio bytes
agent ◄──GET audio────────── MinIO
```

Four properties this layout buys:

- **The worker is a client, not the model.** It speaks HTTP to the TTS server,
  so it does not need torch and does not have to run on the Mac. Moving
  synthesis to another server is changing `MOSS_TTS_URL`.
- **Results stay small.** Temporal payloads warn at 2 MB and fail at 4 MB. A
  minute of 24 kHz mono WAV is about 2.9 MB, so a two-line job would already
  break a workflow that returned bytes. Audio goes to object storage; the
  result carries keys, URLs, sizes and durations.
- **One TTS server, one job.** `max_concurrent_activities=1` over a
  single-thread executor. The server generates serially — a second concurrent
  request only makes both slower. Everything past the one slot is Temporal's
  backlog, which is durable and survives the worker being restarted or moved.
- **The worker needs no open ports.** Temporal workers poll; every connection
  is outbound.

## Running it

The Temporal namespace and the MinIO bucket already exist:

```bash
docker exec opentanuki-temporal temporal operator namespace create \
  --address 127.0.0.1:7233 --namespace moss --retention 72h
# bucket: the minio-init service in ~/Projects/zimage-temporal/deploy
```

Worker, on a machine that can reach the TTS server:

```bash
cd temporal
cp .env.example .env      # set MOSS_S3_SECRET_KEY (or export ZIMAGE_S3_SECRET_KEY)
uv sync
make worker
```

`.env` is read from the working directory or from next to `pyproject.toml`;
real environment variables win over it, so launchd and `FOO=bar moss ...` still
override. `uv run` does not load `.env` by itself — the package does. The S3
settings fall back to `ZIMAGE_S3_*` when `MOSS_S3_*` is unset, because it is
literally the same MinIO with the same credentials and `~/.env` already exports
those names.

No credential is silently defaulted. An empty secret key fails at startup with
a message naming `KEYS.toml`, and `ensure_bucket` treats only a genuine 404 as
"the bucket is missing" — 401 and 403 say the credentials were rejected,
because blindly creating on any error turns a wrong password into a
`SignatureDoesNotMatch` on the wrong operation.

## Using it

```bash
moss submit "Notice. The build finished."                      # returns an id
moss submit --file script.txt --voice handler --prefix ep014   # one clip per line
moss status moss-4f2a1c9d0b77                                  # state + in-clip progress
moss wait   moss-4f2a1c9d0b77                                  # block until done
moss submit "one line" --wait                                  # both at once
moss cancel moss-4f2a1c9d0b77
moss ls --limit 20
moss health                                                    # Temporal + storage + TTS
```

Every command prints JSON, so an agent can parse it without scraping prose:

```json
{
  "id": "moss-verify-001",
  "state": "succeeded",
  "result": {
    "clips": [
      {
        "index": 0,
        "key": "verify/moss-verify-001-000.wav",
        "url": "http://raphael.local:9100/moss/verify/moss-verify-001-000.wav",
        "voice": "handler",
        "bytes": 210988,
        "content_type": "audio/wav",
        "audio_seconds": 4.396,
        "generate_seconds": 31.5
      }
    ],
    "seconds": 31.6,
    "audio_seconds": 4.396
  }
}
```

### Options worth knowing

| Flag | Default | Note |
| --- | --- | --- |
| `--file` | none | One clip per line, `-` for stdin. Blank lines are dropped, because scripts have paragraph breaks. |
| `--voice` | `default` | A reference clip registered on the server (`GET /v1/voices`). Unknown names fall back to the model's own voice. |
| `--model` | empty | Empty means the server's `MODEL_ID`. Naming one swaps the resident model, costing a reload for everything else using that server. |
| `--language` | none | Plain name, e.g. `Japanese`. Omitted, the model infers it from the text. |
| `--format` | `wav` | `wav`, `mp3`, `flac`, `pcm`. `mp3` needs ffmpeg on the *server*. |
| `--prefix` | none | Key prefix in the bucket, e.g. `bedtime/ep014`. |
| `--id` | random | Reusing an id is idempotency: a repeated submission joins the existing run. |

## Design notes

**A 31-line story is one workflow with 31 activities, not 31 workflows.**

The alternative — one workflow per line — was rejected on three counts. A
script is one unit of work to whoever asked for it, so cancelling it should be
one command and not thirty-one; the clips are ordered and belong in one result
array in that order, which a caller would otherwise have to reassemble by
sorting ids; and thirty-one workflows produce thirty-one histories to read when
one line fails, instead of a single history whose failure names the line.

What a workflow-per-line would have bought is parallelism, and there is none to
buy: the server generates serially, so fanning out only moves the queue from
Temporal — where it is durable, visible and cancellable — into someone else's
process.

Splitting the batch into one *activity* per line rather than one activity that
loops (which is what zimage-temporal does for multiple images) is the part
worth copying. Each line gets its own timeout, its own retry budget, and its
own history event, so a batch that fails on line 30 retries line 30 rather than
the whole hour. It also keeps every activity payload small.

The cost accepted: no partial result is returned when the batch fails or is
cancelled. The clips already uploaded do stay in the bucket, and the `status`
query still answers on a closed workflow, which is where you read how far it
got.

**Timeouts are sized off the worst case; the heartbeat does the detecting.**
`start_to_close` is 20 minutes per clip. That is absurd for a median clip and
exactly right for a timeout, because its only job is to be too large to fire on
a healthy run. Liveness is the heartbeat's job: the activity heartbeats every
5s while the HTTP call is outstanding, so a dead worker is noticed in a minute.

The heartbeat carries elapsed seconds rather than a percentage. Generation
reports no intermediate progress, so elapsed time is the only signal that
separates "slow, as usual" from "wedged" — and with a 5x spread on identical
input, a human or an agent needs that number to decide whether to wait.

**Validation happens in the workflow, not the activity.** zimage-temporal
validates inside the activity, which is fine for a single image. With a batch
it is not: a 31-line job whose line 30 is empty should fail in the first second,
not after 29 clips have been generated and paid for. Validation is pure
(lengths and formats — no I/O), so it is safe in a workflow, and it raises a
non-retryable `InvalidRequest`, because retrying identical bad input can only
fail identically. The server's own rejections (400/401/403/404/413/422) map to
the same non-retryable type; 5xx and connection failures stay retryable.

**Cancellation is cooperative, and honest about its limit.** `moss cancel`
requests it; the activity notices at its next heartbeat and stops the batch
there. The clip already in flight keeps generating on the TTS server — it has
no cancel endpoint, and its synthesis runs in a thread that a client disconnect
does not interrupt — and its output is discarded rather than uploaded. What
cancellation actually buys is the remaining clips, which on a 31-line job is
nearly all of the cost. `--terminate` is the hard stop that skips the unwind.

Making that work needed the worker's heartbeat throttle turned down.
Cancellation reaches a running activity only in the *response* to a heartbeat,
and the SDK throttles heartbeats to 30s by default (capped at 60s) — longer
than a whole clip usually takes, so the first cancelled batch generated and
uploaded the in-flight clip anyway, leaving an object in the bucket that no
result referenced. Both throttle intervals are now set to
`MOSS_HEARTBEAT_SECONDS`, which narrows the window from "the whole clip" to
"one heartbeat interval": a clip that finishes generating within a few seconds
of the cancel still uploads, because the cancel has not been delivered yet.
Observed once in verification, two seconds inside the window. The stray object
is harmless and identifiable — its index is at or above the `clips_done` the
status query reports.

The same throttle is why the elapsed-seconds progress is worth reading at all:
at the default it barely moved before a clip was finished.

**The worker refuses to start when the TTS server is down.** Checking `/health`
first and exiting 2 leaves the jobs queued, which is the durable state we
wanted. A worker that starts anyway takes tasks and burns their retry budget on
a server that was never going to answer. `MOSS_TTS_STARTUP_CHECK=0` opts out.

**Why the bucket is anonymous-read.** Same reasoning as zimage: presigned URLs
sign over the `Host` header, so a URL minted for one of this host's names fails
from the other two, and every URL expires. These are generated clips on a
private network. `MOSS_S3_PRESIGN=1` switches.

**Why its own uv project rather than an extra on the root `pyproject.toml`.**
The root project is the TTS server and depends on torch, torchaudio and
torchcodec. This is a client of that server: temporalio, boto3, httpx. Folding
it in would mean `uv sync` pulls several gigabytes of accelerator wheels onto
any machine that only wants to submit a job — including the one running the
worker, which never touches a model. The cost accepted is a second lockfile in
the repo.

**Durations are read, not decoded.** WAV comes from the header and `pcm` from
arithmetic (24 kHz mono s16le). MP3 and FLAC would need a decoder, and pulling
one in to label a number nobody blocks on is not worth the dependency —
`audio_seconds` is `null` for those and says so.

## Testing without the TTS server

`scripts/fake_worker.py` runs the real workflow with silent WAVs, so the
plumbing — serialisation, the status query, heartbeat progress, upload,
returned URL, cancellation — can be exercised on a machine that cannot reach
the Mac.

```bash
make fake-worker
```

Never point it at a queue a real worker is also polling: whichever picks the
task up wins, and half your clips would be silence.
