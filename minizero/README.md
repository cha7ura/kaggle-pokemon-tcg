# minizero — imperfect-information algorithm lab

Pure-NumPy, self-verifying implementations of the methods in the wiki. **Nothing here ships to
Kaggle** — it's the lab where the ideas behind the neural challenger are made runnable and checked.
The shipped agent stays offline/stdlib ([../wiki/00-index.md](../wiki/00-index.md)).

Each algorithm carries its own correctness oracle: it must drive **exploitability → ~0** on a tiny
game. That assertion is the test — no mocks.

## Contents

| Path | What | Maps to | Status |
|------|------|---------|--------|
| `cfr/kuhn.py` | Kuhn poker game tree | [wiki/10](../wiki/10-cfr-and-nash.md) | done |
| `cfr/cfr.py` | vanilla CFR + regret matching (returns the **average** strategy) | [wiki/10](../wiki/10-cfr-and-nash.md), [cfr2013] | done |
| `cfr/exploit.py` | exact best-response exploitability (brute-force pure strategies) | [wiki/10](../wiki/10-cfr-and-nash.md), [ismcts2012] strategy-fusion note | done |
| `cfr/run.py` | watch exploitability → 0 | — | done |
| `cfr/mccfr.py` | external-sampling MCCFR | [wiki/10](../wiki/10-cfr-and-nash.md) | planned (Task 2) |
| `cg/` | ISMCTS + determinization on the real `cg` sim | [wiki/17](../wiki/17-imperfect-info-on-tcg.md) | planned (Docker, Task 3) |
| `rnad/` | tabular R-NaD on Leduc → neural R-NaD | [wiki/12](../wiki/12-deepnash-rnad.md) | planned (Task 4/6) |

Run the working piece:

```bash
python -m minizero.cfr.run            # prints the exploitability curve and learned Nash strategy
python -m pytest minizero/cfr/        # the correctness oracle (exploitability < 1e-2)
```

Citations resolve in [../wiki/papers/sources.md](../wiki/papers/sources.md).
