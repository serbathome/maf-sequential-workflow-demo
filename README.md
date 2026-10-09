# Sequential agent workflow

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12+, then run:

```powershell
uv sync
uv run main.py
```

`uv sync` creates `.venv` and installs the versions in `uv.lock`. The project uses the package index configured in `uv.toml`; if it is unavailable in your environment, change its `index-url` to an accessible PyPI-compatible index and run `uv lock` followed by `uv sync`.

The Azure AI Foundry project endpoint (including the project name) and model deployment names are hard-coded in `main.py`. Update them to match your environment before running. Sign in with `az login` using an account that can access the project and its model deployments. The run prints each agent's response and saves the German, transliterated article under `summaries/`.

## Workflow

`main.py` builds a sequential pipeline: researcher gathers Microsoft Learn facts via MCP; writer drafts an article using word-list tools; editor revises it for a young audience; translator renders it in German, replaces umlauts, and saves it via a skill.
