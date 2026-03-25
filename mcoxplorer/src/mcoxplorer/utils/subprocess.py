from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Standard command execution result."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]


def run_command(command: list[str], cwd: str | Path | None = None) -> CommandResult:
    """Run shell command deterministically and capture output.

    Raises:
        RuntimeError: if command execution fails.
    """

    LOGGER.info("Running command: %s", " ".join(command))
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    result = CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        command=command,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return result
