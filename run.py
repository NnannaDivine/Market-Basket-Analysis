from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
API_HEALTH_URL = "http://127.0.0.1:8000/health"
SOURCE_PATH = ROOT / "src"
sys.path.insert(0, str(SOURCE_PATH))


def wait_for_api(process: subprocess.Popen[bytes], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"API exited before becoming ready (exit code {process.returncode})."
            )
        try:
            with urlopen(API_HEALTH_URL, timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError(f"API did not become ready within {timeout:g} seconds.")


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def launch_services() -> int:
    processes: list[subprocess.Popen[bytes]] = []
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SOURCE_PATH)
        if not existing_pythonpath
        else os.pathsep.join((str(SOURCE_PATH), existing_pythonpath))
    )
    try:
        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "market_basket.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=ROOT,
            env=environment,
        )
        processes.append(api)
        wait_for_api(api)

        dashboard = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(SOURCE_PATH / "market_basket" / "dashboard.py"),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=ROOT,
            env=environment,
        )
        processes.append(dashboard)

        print("API:       http://127.0.0.1:8000 (docs: /docs)", flush=True)
        print("Dashboard: http://127.0.0.1:8501", flush=True)

        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    return return_code
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nStopping the API and dashboard...", flush=True)
        return 0
    finally:
        stop_processes(processes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Market Basket dashboard and recommendation API."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Regenerate rules from data/raw/basket_analysis.csv before starting.",
    )
    args = parser.parse_args()

    if args.train:
        from market_basket.training import train_rules

        rule_count = train_rules()
        print(f"Generated {rule_count:,} association rules.", flush=True)

    try:
        return launch_services()
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"Could not start the project: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
