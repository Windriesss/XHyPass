# DYN/RTO transition experiment

This experiment runs entirely in the Linux Xen dom0. It measures
caller-observed DYN-to-RTO and RTO-to-DYN latency, retry behavior, rollback
outcomes, and empirical transition correctness.

The repository hypervisor includes the local-notification ownership transfer
required by this campaign. During RTO-to-DYN, a Pending-only physical SGI/PPI
is cleared from the RTO pCPU's Redistributor before DYN is published and is
then injected once through the vGIC. The reserved event SGI is treated only as
a proxy: if event-channel state is pending, Xen injects the ordinary virtual
event-channel interrupt instead. Assigned SPI Pending state remains on the
physical path and is handled by Xen after the transition. Physical Active and
ambiguous Active-and-Pending states still reject the transition for retry.

## Prerequisites

- Boot the XHyPass Xen image with a Linux dom0 kernel containing
  `guest/linux/kernel-patches/0001-arm64-xhypass-rto-event-sgi.patch`.
- Pin the RTO dom0 vCPU one-to-one to its pCPU. The examples use vCPU/pCPU 6.
  Remove that pCPU from every other vCPU's hard affinity: Xen rejects
  DYN-to-RTO if the target binding is not exclusive.
- Reserve physical SGI 7 for the RTO event-channel notification.
- Leave `/timer@feae0000` available to dom0 and unclaimed by another driver.
  Its device-tree interrupt `<0 0x121 4>` is GIC INTID 321.
- Build the module and both AArch64 user programs in
  `guest/linux/interrupt_passthrough/`.

## On-target microbenchmark

Build the Linux guest module and benchmark under
`guest/linux/interrupt_passthrough/`. Load the module without entering RTO
automatically:

```bash
insmod ./interrupt_passthrough.ko auto_enter=0 rto_cpu=6 event_sgi=7
taskset -c 6 ./xhypass_transition_bench \
  --iterations 10000 \
  --run-id run-001 \
  --condition idle \
  --output transition_attempts.csv
```

The CSV contains every HVC attempt, including retries. `duration_ns` is the
single-HVC duration, while `request_elapsed_ns` includes all attempts and retry
delays since the start of the current direction.

Store formal data as:

```text
data/<PLATFORM>/transition/<CONDITION>/run_<NNN>/transition_attempts.csv
```

Keep independent runs separate. Do not concatenate them before computing
percentiles.

## Correctness stress

The in-module producer runs on a non-RTO dom0 CPU and continuously exercises:

- Linux cross-CPU calls, delivered as an SGI to the RTO CPU;
- a pinned one-shot high-resolution timer, delivered as a local timer PPI;
- a Xen loopback event channel, whose RTO notification uses reserved SGI 7.
- the RK3588 timer at `feae0000`, which generates a real level-high device
  SPI (GIC INTID 321).

Run the controller on the RTO CPU:

```bash
taskset -c 6 ./xhypass_correctness \
  --iterations 10000 \
  --producer-cpu 0 \
  --platform RK3588 \
  --run-id run-001 \
  --output correctness.json
```

Each notification is acknowledged before the next notification of the same
class is emitted. The JSON therefore checks exact logical completion without
mistaking architecturally permitted interrupt coalescing for loss. Watchdog
timeouts, produced/consumed mismatches, an incomplete transition, or a final
mode other than DYN fail the run.

The SPI source is deliberately platform-specific. The module follows the
Rockchip timer programming sequence: enable the `pclk` and `timer` clocks,
disable and clear the timer, load a one-shot count, then enable it with the
interrupt unmasked. Before the first arm, it resolves the Linux IRQ, verifies
that its GIC hwirq is 321, sets its affinity to the RTO CPU, and verifies the
effective affinity. A busy IRQ, a missing device-tree node, a different hwirq,
or failed affinity verification aborts setup instead of producing a misleading
result. The JSON records Linux IRQ, hwirq, timer rate, affinity CPU, handler CPU
errors, and exact produced/consumed counts.

Reserve the timer for the experiment without letting the built-in Rockchip
clocksource driver claim IRQ 321:

```dts
&rktimer {
    compatible = "xhypass,rk3588-spi-timer";
    status = "okay";
};
```

Do not add `xen,passthrough`: Xen would mark the node disabled in the generated
dom0 device tree. The experiment module recognizes the dedicated compatible
while continuing to use the RK3588/RK3288 timer register layout.

Run an SPI-only pilot first:

```bash
python experiments/transition/run_dom0_campaign.py \
  --mode correctness \
  --correctness-sources spi \
  --correctness-iterations 10000 \
  --spi-delay-us 50 \
  --runs 1 \
  --command-timeout 1800
```

## Automated campaign

The host-side runner deploys the already-built artifacts over SSH and uses
COM10 as the default experiment control and diagnostic channel. It pins dom0
vCPU 6, loads the module in DYN mode, runs latency and correctness stages,
retrieves raw files, and unloads the module on the RTO CPU.

Before collecting data, the runner performs two probe transitions. It rejects
a deployed Xen image that prints an `enter_rto`/`exit_rto` message for every
HVC: synchronous console output inside the measured path both floods COM10 and
invalidates transition latency. Build and deploy the repository Xen source,
whose HVC handler does not contain per-transition success logging.

Correctness sources are run separately in the requested order. The default is
SGI, timer-PPI, and event
channel. A failure therefore identifies the active source instead of losing
all diagnostic context in a combined stress run. The safe default is one
pilot run with 1,000 latency transitions and 100 transitions per correctness
source:

```bash
python experiments/transition/run_dom0_campaign.py
```

Check paths and resolved configuration without connecting to the board:

```bash
python experiments/transition/run_dom0_campaign.py --dry-run
```

Each invocation gets a timestamped directory under
`data/RK3588/transition/campaigns/`. It contains:

- `com10.log`: the complete serial transcript, including Xen/Linux fatal output;
- `events.jsonl`: stage start, completion, and failure timestamps;
- `ssh.log`: deployment and transfer commands;
- before/after snapshots of `dmesg`, `xl dmesg`, `/proc/interrupts`, loaded
  modules, and dom0 vCPU placement;
- the CSV or JSON result for every completed stage.

If a serial timeout or fatal signature occurs, the runner records the failure
and deliberately avoids automatic module unload: issuing cleanup commands to
an unresponsive RTO CPU can obscure the first failure. Recover the board and
inspect that campaign directory before continuing.

After all three pilot sources pass independently, run the formal campaign
explicitly:

```bash
python experiments/transition/run_dom0_campaign.py \
  --runs 10 \
  --iterations 10000 \
  --correctness-iterations 10000
```

To match the completed SGI, timer-PPI, and event-channel campaign, collect the
formal device-SPI data as 100 independent runs, each with 10,000 transition
pairs. The 50-us SPI delay matches the local timer-PPI delay; the remaining
controller parameters are stated explicitly for reproducibility:

```bash
python experiments/transition/run_dom0_campaign.py \
  --mode correctness \
  --correctness-sources spi \
  --correctness-iterations 10000 \
  --runs 100 \
  --timer-delay-us 50 \
  --spi-delay-us 50 \
  --event-timeout-ms 1000 \
  --max-retries 1000 \
  --retry-delay-us 10 \
  --dwell-us 0 \
  --command-timeout 1800
```

## Analyze

```bash
python experiments/transition/analyze_transition.py \
  --platform RK3588 \
  --data-root data/RK3588/transition
```

This writes run-level statistics and a cross-run summary below the data
directory's `analysis/` subdirectory.

The validated raw files used by the paper are already curated under
`data/paper/RK3588/transition/`. Recompute their run-level latency statistics
without accessing the board with:

```bash
python experiments/transition/analyze_transition.py \
  --platform RK3588 \
  --data-root data/paper/RK3588/transition/latency
```

## Correctness records

`correctness_schema.example.json` defines the output schema. Validate completed
records with:

```bash
python experiments/transition/validate_correctness.py \
  data/RK3588/transition/correctness/run_*/correctness.json
```

The result is empirical evidence for the exercised transitions and does not
replace a model proof. Device-SPI coverage is claimed only for records whose
`device-SPI` entry is not skipped and whose `spi_timer.affinity_verified` value
is true.
