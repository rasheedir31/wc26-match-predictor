# data/seed/ - committed reference & offline-fallback data

Small, version-controlled data (unlike `data/raw/`, which is gitignored and
reproduced by the pipeline). Two purposes:

1. **Reference data** the pipeline needs but that isn't a bulk download:
   - `wc26_groups.csv` - the 48-team, 12-group WC26 group stage.
   - `fifa_rankings.csv` - a compact FIFA ranking snapshot (reference / priors / display).

   The group-stage fixture list (round-robin per group) is *generated* deterministically
   from the groups by `wc26.ingest.fixtures` - not committed, to avoid duplication.

2. **Offline fallback** so `make pipeline` and the tests run with no network:
   - `sample_results.csv`, `sample_shootouts.csv` - a small set of international
     matches in the martj42 schema. When the real download is unavailable, ingest
     falls back to these so the pipeline still produces output end to end.

> **Honesty notes.** `sample_results.csv` scores are *illustrative* - when online,
> the ingest layer downloads the full, authoritative
> [martj42/international_results](https://github.com/martj42/international_results)
> dataset and caches it under `data/raw/`. The WC26 **group assignments** here are a
> representative seed; replace `wc26_groups.csv` with the official draw when set.
> Tournament *structure* (12 groups of 4, knockout bracket) is fixed by the format.
