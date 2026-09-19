"""Print actual topology hash and Module 0/4 pytest results; skips are not proof."""

import hashlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    path = Path("configs/topology.yaml")
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}", flush=True)
    status = 0
    for module, test in ((0, "tests/test_catalogue.py"), (4, "tests/test_postgres_logger.py")):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "pytest.xml"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    test,
                    f"--junitxml={report}",
                ],
                capture_output=True,
                text=True,
            )
            lines = result.stdout.strip().splitlines()
            print(f"Module {module}: {lines[-1] if lines else result.stderr.strip()}", flush=True)
            if result.returncode:
                status = 1
                # Pytest tracebacks can include database fixture credentials.
                # Report the outcome without echoing captured connection strings.
                print(
                    f"Module {module}: verification failed (pytest exit {result.returncode})",
                    file=sys.stderr,
                )
            elif not report.exists() or any(
                int(s.get("skipped", 0)) for s in ET.parse(report).iter("testsuite")
            ):
                print(f"Module {module}: skipped checks are not verified evidence", file=sys.stderr)
                status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
