import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import uuid


def Digest(*, value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def FileDigest(*, path: pathlib.Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def Files(*, directory: pathlib.Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): FileDigest(path=path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def ReadJSON(*, path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def Runtime() -> dict:
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())),
    }


def SourceState(*, root: pathlib.Path) -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=root, check=True, capture_output=True)
    # Record changes without copying potentially sensitive source content into published artifacts.
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    hashes = {}
    for name in files.stdout.decode().split("\0"):
        if name and (root / name).is_file():
            hashes[name] = FileDigest(path=root / name)
    return {"revision": revision.stdout.strip(), "status": status.stdout, "files": hashes,
            "diff_sha256": hashlib.sha256(diff.stdout).hexdigest()}


def Timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def WriteJSON(*, path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
