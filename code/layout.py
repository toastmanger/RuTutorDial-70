"""File layout shared by the working repository and the published artifact.

Working repository: corpora, gold and judge logs live under pilot/ (RU corpora in pilot/ru/).
Published artifact: corpora and gold in data/, judge logs in logs/, reports in judges/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if (ROOT / "pilot").is_dir():
    DATA = ROOT / "pilot"
    RU = DATA / "ru"
    LOGS = DATA
    REPORTS = DATA
else:
    DATA = ROOT / "data"
    RU = DATA
    LOGS = ROOT / "logs"
    REPORTS = ROOT / "judges"
