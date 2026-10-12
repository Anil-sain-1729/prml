# Explainable AI-Based Network Intrusion Detection System

Advanced 6th-semester internship starter project, optimized for a 16 GB RAM computer.

## Files
- `notebooks/Explainable_NIDS_Internship.ipynb`: end-to-end notebook.
- `data/synthetic_nids_sample.csv`: 6,000-row **synthetic demo dataset** for verifying notebook execution.
- `requirements.txt`: Python dependencies.
- `ROADMAP.md`: implementation plan and resume guidance.

## Setup (Windows PowerShell)
```powershell
cd Explainable_NIDS_Internship
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```
Open `notebooks/Explainable_NIDS_Internship.ipynb`.

## Important
The included CSV is synthetic, not real network traffic. Do not claim real-world detection performance from it. For final internship evaluation, download CIC-IDS2017/CSE-CIC-IDS2018 data from the official links included in the notebook, rerun the experiment, and report measured metrics honestly.

## RAM guidance
Start with a single CSV and 50k–200k rows. Do not attempt to load multiple 60–100 GB datasets into memory at once. Use Spark/Parquet for scaling in a later phase.
