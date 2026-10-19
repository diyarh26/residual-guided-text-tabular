# Upstream source identity

[multabench.lock.json](multabench.lock.json) pins historical MulTaBench master.
[Provenance](../docs/UPSTREAM.md) records the historical selection, source flows,
paper discrepancies, and license history. No upstream source is vendored.

An optional local checkout at `third_party/MulTaBench/` is ignored; see
[reproduction notes](../docs/REPRODUCIBILITY.md). Do not update the immutable pin
or replace it with current master without an explicit documented decision.
