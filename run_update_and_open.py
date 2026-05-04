#!/usr/bin/env python
"""
Atualiza dados, gera backtest e predições, e então abre o dashboard Streamlit.
Uso: python run_update_and_open.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def get_python_executable() -> str:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def run_step(label: str, command: list[str]) -> None:
    print("=" * 80)
    print(label)
    print(" ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=ROOT, check=True)
    print()


def main() -> int:
    python = get_python_executable()

    steps = [
        ("[1/3] Atualizando base de dados", [python, "get_b3_data.py"]),
        ("[2/3] Gerando backtest", [python, "backtest_model.py"]),
        ("[3/3] Gerando predições para amanhã", [python, "predict_tomorrow.py"]),
    ]

    for label, command in steps:
        run_step(label, command)

    print("=" * 80)
    print("[4/4] Abrindo dashboard Streamlit")
    print("=" * 80)
    subprocess.run([python, "-m", "streamlit", "run", "dashboard.py"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
