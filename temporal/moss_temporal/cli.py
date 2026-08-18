"""Command line client. This is the submit/poll/fetch surface an agent uses.

Output is JSON on stdout so it can be piped into `jq` or parsed by an agent
without scraping prose. Errors go to stderr with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import uuid
from pathlib import Path

from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError
from temporalio.service import RPCError

from .config import settings
from .shared import FORMATS, SynthesizeRequest
from .workflows import SynthesizeSpeechWorkflow


async def connect() -> Client:
    return await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)


def emit(payload) -> None:
    print(json.dumps(payload, indent=2, default=_encode, ensure_ascii=False))


def _encode(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def _parse_meta(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--meta expects key=value, got {pair!r}")
        out[key] = value
    return out


def _lines_from(args) -> list[str]:
    """Positional text, or a file of one line per clip, or stdin.

    Blank lines are dropped rather than rejected, because the thing an agent
    actually has is a script with paragraph breaks in it.
    """
    lines = list(args.text)
    if args.file:
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
        lines += [line.strip() for line in raw.splitlines()]
    return [line for line in lines if line.strip()]


async def _poller_count(client: Client) -> int:
    """How many workers are polling the task queue right now."""
    described = await client.workflow_service.describe_task_queue(
        DescribeTaskQueueRequest(
            namespace=settings.temporal_namespace,
            task_queue={"name": settings.task_queue},
        )
    )
    return len(described.pollers)


async def _heartbeat_progress(client: Client, workflow_id: str):
    """In-clip progress, read out of the running activity's last heartbeat."""
    handle = client.get_workflow_handle(workflow_id)
    try:
        desc = await handle.describe()
        for pending in desc.raw_description.pending_activities:
            if pending.HasField("heartbeat_details") and pending.heartbeat_details.payloads:
                (progress,) = await client.data_converter.decode(pending.heartbeat_details.payloads)
                return dataclasses.asdict(progress) if dataclasses.is_dataclass(progress) else progress
    except Exception:
        return None
    return None


async def cmd_submit(args) -> int:
    lines = _lines_from(args)
    if not lines:
        print("nothing to speak: pass text, or --file PATH (or --file - for stdin)", file=sys.stderr)
        return 1

    client = await connect()
    req = SynthesizeRequest(
        lines=lines,
        voice=args.voice,
        model=args.model,
        response_format=args.format,
        language=args.language,
        prefix=args.prefix,
        metadata=_parse_meta(args.meta),
    )
    workflow_id = args.id or f"moss-{uuid.uuid4().hex[:12]}"

    handle = await client.start_workflow(
        SynthesizeSpeechWorkflow.run,
        req,
        id=workflow_id,
        task_queue=settings.task_queue,
    )

    if not args.wait:
        emit({"id": handle.id, "run_id": handle.result_run_id, "state": "submitted", "clips": len(lines)})
        return 0

    try:
        result = await asyncio.wait_for(handle.result(), timeout=args.timeout)
    except WorkflowFailureError as exc:
        emit({"id": handle.id, "state": "failed", "error": str(exc.cause or exc)})
        return 1
    except asyncio.TimeoutError:
        emit({"id": handle.id, "state": "timeout", "note": f"still running after {args.timeout}s"})
        return 2
    emit({"id": handle.id, "state": "succeeded", "result": result})
    return 0


async def cmd_status(args) -> int:
    client = await connect()
    handle = client.get_workflow_handle(args.id)
    try:
        desc = await handle.describe()
    except RPCError as exc:
        print(f"no such workflow: {args.id} ({exc})", file=sys.stderr)
        return 1

    payload = {
        "id": args.id,
        "temporal_status": desc.status.name if desc.status else None,
        "started_at": str(desc.start_time),
        "closed_at": str(desc.close_time) if desc.close_time else None,
    }
    if desc.status == WorkflowExecutionStatus.RUNNING:
        payload["workers"] = await _poller_count(client)
        # A query is answered by a worker, not by the server. With none polling
        # -- the usual case while the TTS box is asleep -- asking would block
        # until the RPC times out, so say what is true instead of hanging.
        if payload["workers"]:
            payload["query"] = await _query(handle)
            progress = await _heartbeat_progress(client, args.id)
            if progress:
                payload["progress"] = progress
        else:
            payload["query"] = {
                "state": "queued",
                "note": f"no worker is polling {settings.task_queue}; the job waits until one is",
            }
    elif desc.status == WorkflowExecutionStatus.COMPLETED:
        payload["result"] = await handle.result()
    else:
        # Closed, but not successfully. The query still answers on a closed
        # workflow, and after a cancel it is the only place the partial clip
        # list is written down.
        if await _poller_count(client):
            payload["query"] = await _query(handle)
    emit(payload)
    return 0


async def _query(handle):
    try:
        return await asyncio.wait_for(handle.query("status"), timeout=10)
    except (RPCError, asyncio.TimeoutError):
        return {"state": "unknown", "note": "no worker answered the status query"}


async def cmd_wait(args) -> int:
    client = await connect()
    handle = client.get_workflow_handle(args.id)
    try:
        result = await asyncio.wait_for(handle.result(), timeout=args.timeout)
    except WorkflowFailureError as exc:
        emit({"id": args.id, "state": "failed", "error": str(exc.cause or exc)})
        return 1
    except asyncio.TimeoutError:
        emit({"id": args.id, "state": "timeout"})
        return 2
    emit({"id": args.id, "state": "succeeded", "result": result})
    return 0


async def cmd_cancel(args) -> int:
    client = await connect()
    handle = client.get_workflow_handle(args.id)
    if args.terminate:
        await handle.terminate(reason="cancelled from moss cli")
    else:
        # Cancel is cooperative: the activity notices at its next heartbeat and
        # stops the batch there. The clip already in flight finishes on the TTS
        # server -- it has no cancel endpoint -- and is discarded.
        await handle.cancel()
    emit({"id": args.id, "state": "terminated" if args.terminate else "cancel_requested"})
    return 0


async def cmd_ls(args) -> int:
    client = await connect()
    rows = []
    query = args.query or "WorkflowType = 'SynthesizeSpeech'"
    async for wf in client.list_workflows(query, limit=args.limit):
        rows.append(
            {
                "id": wf.id,
                "status": wf.status.name if wf.status else None,
                "started_at": str(wf.start_time),
                "closed_at": str(wf.close_time) if wf.close_time else None,
            }
        )
    emit(rows)
    return 0


async def cmd_health(args) -> int:
    from . import storage, tts

    out: dict = {
        "temporal_address": settings.temporal_address,
        "namespace": settings.temporal_namespace,
        "task_queue": settings.task_queue,
        "tts_url": settings.tts_url,
        "s3_endpoint": settings.s3_endpoint,
        "bucket": settings.s3_bucket,
    }
    try:
        client = await connect()
        out["workers"] = await _poller_count(client)
        out["temporal"] = "ok"
    except Exception as exc:  # noqa: BLE001 - health reports, never raises
        out["temporal"] = f"error: {exc}"
    try:
        storage.client().head_bucket(Bucket=settings.s3_bucket)
        out["storage"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["storage"] = f"error: {exc}"
    try:
        out["tts"] = tts.health()
    except Exception as exc:  # noqa: BLE001
        out["tts"] = f"error: {exc}"
    ok = out.get("temporal") == "ok" and out.get("storage") == "ok" and isinstance(out.get("tts"), dict)
    emit(out)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moss", description="Submit MOSS-TTS speech jobs to Temporal.")
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="queue a synthesis job (one clip per line)")
    submit.add_argument("text", nargs="*", help="one line of speech per argument")
    submit.add_argument("--file", default=None, help="read lines from a file, or - for stdin")
    submit.add_argument("--voice", default="default", help="reference clip name on the server")
    submit.add_argument("--model", default="", help="empty = the server's own default model")
    submit.add_argument("--language", default=None, help='plain name, e.g. "Japanese"')
    submit.add_argument("--format", default="wav", choices=list(FORMATS))
    submit.add_argument("--prefix", default="", help="key prefix inside the bucket")
    submit.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    submit.add_argument("--id", default=None, help="workflow id; reusing one is how you get idempotency")
    submit.add_argument("--wait", action="store_true", help="block until the audio exists")
    submit.add_argument("--timeout", type=float, default=3600)
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="one job's state and progress")
    status.add_argument("id")
    status.set_defaults(func=cmd_status)

    wait = sub.add_parser("wait", help="block until a job finishes")
    wait.add_argument("id")
    wait.add_argument("--timeout", type=float, default=3600)
    wait.set_defaults(func=cmd_wait)

    cancel = sub.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("id")
    cancel.add_argument("--terminate", action="store_true", help="hard stop, no unwind")
    cancel.set_defaults(func=cmd_cancel)

    ls = sub.add_parser("ls", help="recent jobs")
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--query", default=None, help="Temporal list filter")
    ls.set_defaults(func=cmd_ls)

    health = sub.add_parser("health", help="check Temporal, worker count, storage and the TTS server")
    health.set_defaults(func=cmd_health)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(asyncio.run(args.func(args)))


if __name__ == "__main__":
    main()
