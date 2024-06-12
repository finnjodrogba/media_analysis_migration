# Bachelor Arbeit Finn Mittnacht

Sentiment analysis

## Installation

Creating with virtual env

here for Mac, assuming virtual environment is named ba_fmitt

```bash
python3.11 -m venv ba_fmitt
source ba_fmitt/bin/activate
pip install jupyter
#ensure path to jupyter is set 
deactivate
source ba_fmitt/bin/activate
```

Following folders should exist:

- data/sentiws (with SentiWS_v2.0_Negative.txt,SentiWS_v2.0_Positive.txt)
- helper (with some python py files)

- default: NewsArticles (Folder for pdf news files)
- default: ShortNewsArticles (Folder for testing, with some small pdf files)
- korpus_calculated.csv: File with all calculated values

## Usage

run jupyter-notebook or jupyter-lab

run BuildDataFrame.ipynb in Jupyter

## Problems

Downloading ntlk

```bash
sh "/Applications/Python 3.11/Install Certificates.command
```
