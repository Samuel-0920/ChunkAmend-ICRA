# ChunkAmend Reproduction

Reproduction materials for simulation and UF850 hardware experiments.

## Status

This repository currently contains the reproduction checklist only. It is not yet an executable or validated reproduction release. Algorithm snapshots, dependency locks, portable launch commands and evidence manifests have not yet been imported.

## Version boundaries

- Historical simulation: C209, asynchronous raw-ready execution.
- Formal UF850 experiments: V004 asynchronous Q6 with reserved commit boundaries, H15 / K5, target 30 Hz, nominal three-slot correction budget.
- Existing public `Samuel-0920/chunkamend` repository: its README describes a stride-five configuration and a synchronous d=0 RTC comparator. Its relationship to the historical simulation release must be established before importing it.
- Hardware demonstration wrappers are separate from the formal evaluation protocol.

See [the reproduction checklist](docs/REPRODUCIBILITY_CHECKLIST.md).
