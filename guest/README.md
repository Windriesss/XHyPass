# Guest support

The `linux/interrupt_passthrough/` directory contains the Linux dom0 control
module and transition/correctness tools for XHyPass. They invoke the
caller-local XHyPass HVC interface to switch the current vCPU between DYN and
RTO execution. `linux/kernel-patches/` connects the reserved RTO event SGI to
the Linux Xen event-channel upcall path.

The hypervisor must be built from `../hypervisor/`, and the selected guest vCPU
must be exclusively pinned to its paired pCPU before the module requests RTO.
