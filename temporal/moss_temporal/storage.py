"""S3-compatible object storage (MinIO on the raphael host).

The same MinIO that serves zimage-temporal, a different bucket. Lifted from
that project, including the two mistakes it already paid for: an empty secret
must fail at startup naming the file the real value lives in, and `ensure_bucket`
must treat only a genuine 404 as "missing".
"""

from __future__ import annotations

import functools

import boto3
from botocore.config import Config

from .config import settings


@functools.lru_cache(maxsize=1)
def client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        # MinIO published on a port with no wildcard DNS: virtual-host style
        # addressing cannot resolve, so path style is mandatory.
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 5, "mode": "standard"}),
    )


class StorageAuthError(RuntimeError):
    """Credentials the object store rejected. Not something a retry fixes."""


def ensure_bucket() -> None:
    """Create the bucket if the compose init container has not already.

    Only a genuine 404 means "missing". A 403 means the credentials were
    rejected, and blindly creating on any error turned a wrong password into a
    confusing `SignatureDoesNotMatch` on CreateBucket — an error about the
    wrong operation entirely.
    """
    from botocore.exceptions import ClientError

    if not settings.s3_secret_key:
        raise StorageAuthError(
            "MOSS_S3_SECRET_KEY is empty (and ZIMAGE_S3_SECRET_KEY, its accepted "
            "fallback, is unset too). Set it in .env next to pyproject.toml "
            "(see .env.example) or in the environment; the value is in KEYS.toml "
            "under [zimage_minio]."
        )

    c = client()
    try:
        c.head_bucket(Bucket=settings.s3_bucket)
        return
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status in (401, 403):
            raise StorageAuthError(
                f"{settings.s3_endpoint} rejected the credentials for access key "
                f"{settings.s3_access_key!r} (HTTP {status}). Check MOSS_S3_ACCESS_KEY "
                f"and MOSS_S3_SECRET_KEY against KEYS.toml, [zimage_minio]."
            ) from exc
        if status != 404:
            raise

    c.create_bucket(Bucket=settings.s3_bucket)


OWNER_KEY = "moss-owner"


class KeyCollision(RuntimeError):
    """The key is already taken by a different run. Not something a retry fixes."""


def owner_of(key: str) -> str | None:
    """The run that wrote `key`; `""` if something is there untagged; None if
    nothing is.

    The empty string matters. Collapsing "no object" and "an object nobody
    tagged" into None would wave through exactly the objects written before
    this guard existed, or by anything else sharing the bucket -- the case the
    guard is for.
    """
    from botocore.exceptions import ClientError

    try:
        head = client().head_object(Bucket=settings.s3_bucket, Key=key)
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            return None
        raise
    # boto3 lowercases user metadata keys on the way out.
    return head.get("Metadata", {}).get(OWNER_KEY, "")


def put(key: str, data: bytes, content_type: str, owner: str) -> str:
    """Upload and return the URL a consumer should fetch.

    Conditional, not last-writer-wins. Object keys are derived from the
    workflow id, and a workflow id can come back: Temporal's default reuse
    policy starts a *new* run under an id whose previous run has closed. A bare
    put_object would then silently overwrite the first run's audio while its
    history still advertised those URLs, and nothing would error.

    `IfNoneMatch="*"` makes the write fail instead (MinIO has supported it
    since RELEASE.2024-10-29). On 412 the owner metadata decides which case it
    is: our own earlier attempt, which is an idempotent success, or somebody
    else's object, which is a collision this run must not paper over.
    """
    from botocore.exceptions import ClientError

    try:
        client().put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={OWNER_KEY: owner},
            IfNoneMatch="*",
        )
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status not in (409, 412):
            raise
        # An activity retry after a crash between upload and return finds its
        # own object here. Refusing that would turn every transient failure
        # into a permanent one, under exactly the conditions no hand-written
        # test covers.
        existing = owner_of(key)
        if existing == owner:
            return url_for(key)
        raise KeyCollision(
            f"{key} already exists in bucket {settings.s3_bucket}, written by "
            f"{existing or 'an untagged writer'} rather than {owner}. Submit with a "
            f"fresh --id, or a --prefix that separates the two."
        ) from exc
    return url_for(key)


def url_for(key: str) -> str:
    if settings.s3_presign:
        # Presigned URLs are signed over the Host header, so they only work
        # when the consumer uses the same endpoint the signature was made for.
        return client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=settings.s3_presign_ttl,
        )
    return f"{settings.public_endpoint.rstrip('/')}/{settings.s3_bucket}/{key}"
