# Linux RTO control module

This loadable module invokes the XHyPass private Arm HVC ABI from the selected
Linux vCPU. Loading the module requests DYN-to-RTO; unloading it requests
RTO-to-DYN.

## Requirements

- An AArch64 Linux dom0 running on the XHyPass-enabled Xen hypervisor.
- The dom0 event-SGI patch in `../kernel-patches/`.
- The selected guest vCPU exclusively pinned to the paired physical CPU.
- A physical SGI reserved for XHyPass event-channel notification and unused by
  Xen or other platform software.
- A configured Linux kernel source or build tree for the guest kernel.

## Build

For a module targeting the currently running kernel:

```bash
make
```

For another prepared kernel tree, override `KERNEL_SRC`. Relative paths are
supported; for example, if the kernel tree is next to this repository:

```bash
make KERNEL_SRC=../../../../rockchip-kernel \
  ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu-
```

Replace `aarch64-linux-gnu-` with the prefix of the toolchain used to build the
guest kernel. Building natively inside an AArch64 guest normally requires only
`KERNEL_SRC`.

The build produces `interrupt_passthrough.ko`, `xhypass_transition_bench`, and
`xhypass_correctness` locally. Build products are ignored by the repository.

## Enter and leave RTO

Run module insertion on the target CPU so that the caller-local HVC is issued
by the intended vCPU:

```bash
taskset -c 6 insmod ./interrupt_passthrough.ko rto_cpu=6 event_sgi=7
```

Return the vCPU to DYN before changing its CPU affinity or reclaiming its paired
pCPU:

```bash
rmmod interrupt_passthrough
```

For repeated userspace-controlled transitions, load without automatically
entering RTO:

```bash
taskset -c 6 insmod ./interrupt_passthrough.ko \
  auto_enter=0 rto_cpu=6 event_sgi=7
```

The defaults are `rto_cpu=6`, `event_sgi=7`, `exit_retry_count=1000`, and
`exit_retry_delay_ms=1`. Platform deployments must select values consistent
with their Xen vCPU pinning and SGI reservation. If RTO exit remains blocked by
an Active interrupt after all retries, the module logs an emergency message;
the system may still be in RTO and must not reassign the paired pCPU.

## ABI

The module exposes `/dev/xhypass` to the two experiment programs and uses
`HVC #0x4a48`. Register `x0` carries operation 12 (enter RTO)
or 13 (exit RTO), and `x1` carries the reserved event SGI on entry. The return
code is delivered in `x0`.

The module is licensed under GPL-2.0.

The correctness tool accepts `--source sgi`, `--source timer`, `--source
event`, or `--source spi`. The SPI source uses the dedicated RK3588 timer node
described in `../../../experiments/transition/README.md` and verifies GIC hwirq
321 and effective affinity to the RTO CPU before arming it. Run sources
separately while diagnosing a platform; this keeps the last completed stage
and the failing interrupt class unambiguous.
