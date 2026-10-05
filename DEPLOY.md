# Deploying to Hugging Face Spaces

## 1. Rotate your HF token first

Your token in `.env` has been exposed in plaintext — create a new one at
https://huggingface.co/settings/tokens (read + inference permissions) and update `.env`.
Never commit `.env` (already in `.gitignore`).

## 2. Create the Space

- Go to https://huggingface.co/new-space
- Owner: your username, Space name: e.g. `walmart-retail-orchestrator`
- SDK: **Streamlit**, Hardware: CPU basic, Public/Private as needed

## 3. Add secrets (Settings → Variables and secrets)

| Name              | Value                                             |
|-------------------|---------------------------------------------------|
| `HF_TOKEN`        | your new token (scope: read + inference)          |
| `HF_CHAT_MODEL`   | `meta-llama/Llama-3.1-8B-Instruct` (optional)     |
| `TOP_K`           | `4` (optional)                                    |

Note: `meta-llama/Llama-3.1-8B-Instruct` is gated — accept the license on the model
page with the same account as the token.

## 4. Push the code

```bash
cd D:\Opencode
git init
git remote add space https://huggingface.co/spaces/<your-username>/walmart-retail-orchestrator
git add app.py main.py README.md requirements.txt src/
git commit -m "Deploy to Spaces"
git push space main
```

Or with the CLI:

```bash
huggingface-cli login
huggingface-cli repo create walmart-retail-orchestrator --type space --space_sdk streamlit
git remote add space https://huggingface.co/spaces/<your-username>/walmart-retail-orchestrator
git push space main
```

## Notes

- `data/` is gitignored — the SQLite vector store is empty on the Space and
  auto-ingests from `pavankm96/KB` on the first request (expect a slow first load).
- The README.md YAML frontmatter (`sdk: streamlit`, `app_file: app.py`) is already set.
- Space storage is ephemeral: KB re-ingestion on app restart is expected.
