from pathlib import Path
import subprocess
import sys
import os
from dungeon.generator import generate, solution
from dungeon.entities import DungeonConfig, ACTIONS

def test_human_cli_oracle_escape():
    root = Path(__file__).resolve().parents[1]
    actions = solution(generate(42,DungeonConfig()))
    result = subprocess.run([sys.executable,"play.py","--seed","42"],cwd=root,
                            input="\n".join(ACTIONS[a] for a in actions)+"\n",text=True,
                            capture_output=True,env={**os.environ,"PYTHONIOENCODING":"utf-8"},encoding="utf-8",timeout=30)
    assert result.returncode == 0, result.stderr
    assert "'success': True" in result.stdout
