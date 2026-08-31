# XHyPass artifact

This repository contains the XHyPass Xen implementation, experiment automation,
workload generators, published measurements, and the scripts used to reproduce
the evaluation figures.

## Repository layout

```text
hypervisor/              Xen 4.20 source tree with the XHyPass RTO vCPU changes
experiments/automation/  RK3588 and E2000Q experiment controllers
experiments/loadgen/     Composite HTTP workload generator
experiments/workloads/   Guest-side workload scripts
data/RK3588/             Published RK3588 measurements
data/E2000Q/             Published E2000Q measurements
plots/                   Figure and analysis scripts
plots/output/            Reproduced figures
reproduce_figures.py     One-command figure reproduction entry point
```

All repository references are relative to the repository root. Hardware boot
images, board credentials, serial-port assignments, and deployment directories
are site-specific and are intentionally not included.

## Reproduce the paper figures

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_figures.py
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. The command produces these PDFs in
`plots/output/`:

- `motivation_tradeoff.pdf`
- `interrupt_latency_runmax.pdf`
- `interrupt_latency_tail_percentiles.pdf`
- `cyclictest_runmax.pdf`
- `rk3588_nn_metrics_scatter_2x3.pdf`
- `rk3588_nn_comprehensive_4x3.pdf`

The plotting scripts select five completed formal runs per published condition.
Interrupt-latency percentiles are computed per run before aggregation; samples
from different repetitions are not merged.

## Hypervisor source

`hypervisor/` is based on Xen 4.20.4 and includes the XHyPass Arm RTO execution
mode. Build prerequisites and commands follow the Xen documentation in
`hypervisor/README`, `hypervisor/INSTALL`, and `hypervisor/docs/`.

The Xen source retains its original copyright notices and license files under
`hypervisor/COPYING` and `hypervisor/LICENSES/`.

## Experiments

The automation supports bare metal, Jailhouse, Xen Credit2, Xen Null, their
native-WFx variants, and XHyPass on the two evaluated Arm platforms. See
`experiments/README.md` for the host and target setup. Published data are kept
separate from site-specific boot artifacts so figure reproduction does not
require access to the experiment hardware.
