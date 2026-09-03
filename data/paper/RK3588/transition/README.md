# RK3588 transition paper data

This directory contains the formal raw data used for the XHyPass transition
evaluation. The four correctness sources contain 100 independent runs with
10,000 DYN/RTO transition pairs per run. The latency dataset contains 30
independent idle runs with 100,000 pairs per run.

## Layout

- `correctness/{sgi,timer,event,spi}/run_NNN/correctness.json`
- `latency/idle/run_NNN/transition_attempts.csv`
- `latency/analysis/`: reproducible run-level and cross-run statistics
- `paper_data_manifest.json`: parameters and source campaign identifiers
- `validation_summary.json`: aggregate validation results

Validate correctness records from the repository root:

```bash
python experiments/transition/validate_correctness.py \
  data/paper/RK3588/transition/correctness/*/run_*/correctness.json
```

Regenerate latency statistics:

```bash
python experiments/transition/analyze_transition.py \
  --platform RK3588 \
  --data-root data/paper/RK3588/transition/latency
```

The original diagnostic campaigns remain under
`data/RK3588/transition/campaigns/` and are not duplicated here.

The SGI, timer-PPI, and event-channel records come from a campaign whose
overall status is `failed` only because its separate SPI run 100 was not
started after a serial-control timeout. All 100 selected runs for each of the
three sources are present and independently validated. The formal SPI records
come exclusively from the completed SPI-only campaign.

Latency statistics use the caller-observed complete request latency stored in
`request_elapsed_ns`, including any retry time. `duration_ns` records an
individual ioctl attempt and is retained only as raw diagnostic data.
