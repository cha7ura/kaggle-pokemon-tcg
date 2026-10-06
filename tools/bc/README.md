# Behaviour-cloning pipeline

This directory is the production path from replay rows to an exported agent.

## Pipeline

```text
replay parquet
  -> build_dataset.py
  -> ragged tensor cache
  -> train.py
  -> checkpoint
  -> export_agent.py
  -> agents/<name>/
  -> build_submission.py
  -> validated submission archive
```

## Core files

| File | Responsibility |
| --- | --- |
| `rows.py` | Build the same semantic row from parquet or a live observation |
| `cards.py` | Build card and attack features from the licensed SDK at runtime |
| `encoder.py` | Encode global state, board/deck tokens, logs, and options |
| `build_dataset.py` | Filter replay rows and write a ragged tensor cache |
| `dataset.py` | Memory-mapped cache reader and batch collation |
| `model.py` | Transformer state encoder and legal-option scorer |
| `train.py` | Optimization, validation, checkpoint metadata, and metrics |
| `agent_main.py` | Guarded runtime shell copied into exported agents |
| `export_agent.py` | Freeze source, weights, deck, configuration, and fingerprint |
| `selftest.py` | Verify live and training encoders agree |
| `validate_resolution.py` | Verify option-to-card resolution against replay logs |

## Reproduction shape

```bash
python tools/bc/build_dataset.py --sdk <sdk> --out data/tensors/<run> \
  --with-logs --deck-tokens

python tools/bc/train.py --data data/tensors/<run> --sdk <sdk> \
  --out runs/<run> --dim 192 --layers 6 --heads 4 --ff 768 --decoder

python tools/bc/export_agent.py --checkpoint runs/<run>/best.pt \
  --agent agents/<name> --deck decks/<deck>.csv
```

Run encoder parity and resolution checks before evaluation, then build and inspect the final
archive with `tools/build_submission.py` and `tools/check_submission.py`.

## Design invariants

- Training and inference use the same encoder source.
- Variable-length tokens and options remain ragged on disk and are padded per batch.
- Illegal or padded options never receive an actionable score.
- The checkpoint records the encoder fingerprint and architecture flags.
- Exported agents fail safe to a legal action when model loading or inference is uncertain.
