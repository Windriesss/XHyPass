# XHyPass artifact

This repository contains the XHyPass Xen implementation, experiment automation,
workload generators, published measurements, and the scripts used to reproduce
the evaluation figures.

## Repository layout

```text
hypervisor/              Xen 4.20 source tree with the XHyPass RTO vCPU changes
guest/linux/             Linux guest control module for DYN--RTO transitions
experiments/automation/  RK3588 and E2000Q experiment controllers
experiments/transition/  Dom0-only transition latency and correctness campaign
experiments/loadgen/     Composite HTTP workload generator
experiments/workloads/   Guest-side workload scripts
data/RK3588/             Published measurements for the original six figures
data/E2000Q/             Published measurements for the original six figures
data/paper/RK3588/       Curated transition validation data
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

`hypervisor/` is based on Xen's stable 4.20 branch (the imported snapshot
identifies itself as 4.20.5-pre) and includes the XHyPass Arm RTO execution
mode described in the paper. The implementation keeps DYN and RTO as the two
stable vCPU modes and uses internal transition states plus a per-vCPU lock to
serialize interrupt producers with ownership transfer.

Before accepting DYN-to-RTO, Xen validates the scheduler-side exclusive
binding: the requesting vCPU's hard affinity must contain exactly its current
pCPU, and no other sched unit that can run in that cpupool may include the same
pCPU in its hard affinity. A failed check leaves the vCPU in DYN mode. This is
a transition-time validation; the deployment must not change vCPU affinities
while RTO is active.

On DYN-to-RTO entry, supported pending vGIC notifications are materialized on
the physical side before their virtual representation is removed. On
RTO-to-DYN exit, assigned SPI Pending state remains physical for Xen's ordinary
IRQ path, while pending local SGI/PPI notifications are removed from the
Redistributor and reinjected through the vGIC after DYN is published. The
reserved event SGI is not exposed as a guest SGI on exit; Xen reconstructs the
ordinary event-channel interrupt from the shared event-channel pending state.
Active or otherwise ambiguous ownership still causes a retry without changing
the committed source mode.

Build prerequisites and commands follow the Xen documentation in
`hypervisor/README`, `hypervisor/INSTALL`, and `hypervisor/docs/`.

The Xen source retains its original copyright notices and license files under
`hypervisor/COPYING` and `hypervisor/LICENSES/`.

## Guest control module

`guest/linux/interrupt_passthrough/` contains the loadable Linux module and
userspace tools used to request DYN-to-RTO and RTO-to-DYN transitions through
the XHyPass Arm HVC ABI. `guest/linux/kernel-patches/` contains the small dom0
kernel change that dispatches the reserved RTO event SGI to Xen's event-channel
upcall handler.
The module must execute on the vCPU that has already been exclusively pinned to
the paired pCPU. See the module README for build parameters, loading commands,
and deployment requirements.

The transition/correctness artifact runs entirely inside Xen dom0 Linux. See
`experiments/transition/README.md` for the on-target commands, automated SSH
campaign, raw-data schema, validator, and analysis workflow.

The supplemental transition-validation dataset is under
`data/paper/RK3588/transition/`. It contains 100 independent correctness runs
for each of SGI, timer-PPI, event-channel, and device-SPI, plus 30 independent
idle transition-latency runs. Its manifest records the experiment parameters,
and the validation summary records aggregate outcomes. Board-local serial logs
and diagnostic campaigns are deliberately excluded from the public artifact.

## Experiments

The automation supports bare metal, Jailhouse, Xen Credit2, Xen Null, their
native-WFx variants, and XHyPass on the two evaluated Arm platforms. See
`experiments/README.md` for the host and target setup. Published data are kept
separate from site-specific boot artifacts so figure reproduction does not
require access to the experiment hardware.
