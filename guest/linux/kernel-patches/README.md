# Linux dom0 event-SGI patch

During RTO, Xen replaces the ordinary virtual event-channel interrupt with an
equivalent physical notification. On this platform that notification is
non-secure SGI 7. Linux must therefore dispatch SGI 7 to the Xen event-channel
upcall handler.

Apply the patch to the Linux 5.10 dom0 kernel tree before building it:

```bash
git apply /path/to/XHyPass-open-source/guest/linux/kernel-patches/0001-arm64-xhypass-rto-event-sgi.patch
```

The Arm GICv3 driver used by the RK3588 kernel already allocates all eight
non-secure SGIs and calls `set_smp_ipi_range(..., 8)`. If a different kernel
allocates only `NR_IPI` SGIs, change that allocation to eight as part of the
port. The patch makes the eighth Linux IPI (`IPI_XHYPASS_EVENT`) invoke
`xen_hvm_evtchn_do_upcall()`.

The module and experiment commands must use `event_sgi=7`. The same SGI must
remain outside Xen's statically assigned SGI range and must not be registered
by another Xen subsystem.
