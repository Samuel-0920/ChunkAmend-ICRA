# Implementation notes

This document identifies the behavior represented by the released source files. The domain settings are recorded in [`configs/`](../configs/).

## Correction core

[`make_corrector`](../chunkamend/core.py) selects the Q6 configuration for `simulation` or `hardware`. Here, **Q6** denotes a local response model with six command inputs and three translational-response outputs; it is not the action horizon.

The core fits its response model from completed command–measurement pairs, proposes a bounded amendment, and checks the candidate against its objective and terminal constraints. Rejected candidates retain the raw plan. The selected translation candidate has length 3, strength 0.7, and a correction cap of 0.06 in normalized controller-action coordinates. The rotational return and writable window span 5 commands; gripper values and the remaining suffix are preserved.

The response model is updated from execution feedback. The policy network is not trained or updated by this module.

## Asynchronous adoption

### Simulation

[`CommonExecutor`](../chunkamend/simulation/reference_executor.py) runs inference outside the control lock and makes the raw result available while correction is computed. A correction can be adopted if its generation and execution cursor remain valid. If the next control tick arrives first, the ready raw result is published; the late correction cannot overwrite it. A stale correction is discarded rather than applied at a different boundary.

### Hardware

[`ReservedQExecutor`](../chunkamend/hardware/executor.py) uses separate inference and correction workers. Once a raw result is accepted, it reserves a future boundary and continues the old action buffer. The nominal budget is 3 commands, bounded by the minimum issued prefix and remaining horizon. At the boundary, it commits a prepared correction or the exact raw fallback.

The correction anchor is the last old-plan target before that boundary. The response fit uses only already-completed feedback; the planned anchor is not treated as a future measurement.

## Action and feedback conventions

The numerical correction core consumes controller-action histories and measured TCP positions in metres. It does not directly fit raw joint-angle commands as Cartesian displacement inputs.

The hardware boundary expects normalized model chunks of shape `15 × 32`, absolute physical joint/gripper chunks of shape `15 × 7`, and a seven-dimensional query state. [`UF850ActionCoordinates`](../chunkamend/hardware/coordinates.py) converts query-relative joint offsets to absolute targets and rebases an older RTC prior into the current query coordinates.

[`FeedbackSnapshot`](../chunkamend/hardware/contracts.py) holds `N × 7` normalized and physical controller-action histories and `(N + 1) × 3` measured TCP positions, with an optional joint anchor. Model inference, kinematic conversion, sensor acquisition, and actual robot I/O belong to the surrounding application; the full application is not included in this release.
