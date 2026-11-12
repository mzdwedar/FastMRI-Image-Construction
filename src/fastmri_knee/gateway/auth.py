"""API-key auth: SHA-256 hashes from a mounted file, compared in constant time.

Clients send `X-API-Key: <key>`. The file holds one hex digest per line (blank lines and `#` comments ignored),
so a leaked file does not leak usable keys. `python -m fastmri_knee.gateway.auth` mints a key.
"""

import hashlib
import hmac
import secrets
import sys
from pathlib import Path

from fastapi import Request

from fastmri_knee.gateway.errors import ProblemError
from fastmri_knee.gateway.settings import Settings

HEADER = "X-API-Key"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_key_hashes(path: str) -> list[bytes]:
    """Read the hashes file. Raises `ValueError` (a startup error) if it is missing, empty or malformed."""
    try:
        text = Path(path).read_text()
    except OSError as e:
        raise ValueError(f"cannot read API keys file {path!r}: {e.strerror}") from None
    hashes: list[bytes] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        if len(line) != 64 or any(c not in "0123456789abcdef" for c in line):
            raise ValueError(f"{path}:{n}: not a SHA-256 hex digest")
        hashes.append(line.encode())
    if not hashes:
        raise ValueError(f"API keys file {path!r} has no keys; run `make api-key`")
    return hashes


def key_hashes_for(settings: Settings) -> list[bytes] | None:
    """The accepted hashes, or None when auth is disabled (dev only, enforced by `Settings`)."""
    return None if settings.auth_disabled else load_key_hashes(settings.api_keys_file)


def _matches(presented: str, hashes: list[bytes]) -> bool:
    digest = hash_key(presented).encode()
    ok = False
    for h in hashes:  # no early exit: time does not depend on which key matched
        ok |= hmac.compare_digest(digest, h)
    return ok


async def require_api_key(request: Request) -> None:
    hashes: list[bytes] | None = request.app.state.key_hashes
    if hashes is None:
        return
    presented = request.headers.get(HEADER)
    if not presented or not _matches(presented, hashes):
        raise ProblemError(401, "Missing or invalid API key", {"WWW-Authenticate": "ApiKey"})


def main(argv: list[str]) -> int:
    """Mint a key: print it once, append its hash to the keys file (default `secrets/api_keys`)."""
    path = Path(argv[0] if argv else "secrets/api_keys")
    key = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(hash_key(key) + "\n")
    path.chmod(0o600)
    print(f"API key (shown once, store it now): {key}\nHash appended to {path}", file=sys.stderr)
    print(key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
