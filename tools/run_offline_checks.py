"""Run manual checks while denying credential-file reads and external networking."""
from pathlib import Path
import os
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def offline_guard(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = Path(os.fsdecode(args[0])).name.casefold()
        if name == ".env" or name.startswith(".env.") or name == ".modal.toml":
            raise PermissionError("Credential-file access is prohibited during verification")
    if event in ("socket.connect", "socket.getaddrinfo"):
        address = args[1] if event == "socket.connect" else args[0]
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1", None):
            raise PermissionError("External networking is prohibited during verification")


if __name__ == "__main__":
    sys.addaudithook(offline_guard)
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
    if len(sys.argv) > 1 and sys.argv[1] == "artifacts":
        runpy.run_path(str(ROOT / "tools/check_artifacts.py"), run_name="__main__")
    else:
        suite = unittest.TestLoader().discover(str(ROOT / "tests"), pattern="test*.py")
        raise SystemExit(not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful())
