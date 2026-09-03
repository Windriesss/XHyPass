/* SPDX-License-Identifier: GPL-2.0 */
#ifndef XHYPASS_IOCTL_H
#define XHYPASS_IOCTL_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define XHYPASS_DEVICE_PATH "/dev/xhypass"
#define XHYPASS_IOCTL_ABI_VERSION 2U

#define XHYPASS_SELFTEST_SGI           (1U << 0)
#define XHYPASS_SELFTEST_TIMER_PPI     (1U << 1)
#define XHYPASS_SELFTEST_EVENT_CHANNEL (1U << 2)
#define XHYPASS_SELFTEST_TIMER_SPI     (1U << 3)
#define XHYPASS_SELFTEST_ALL (XHYPASS_SELFTEST_SGI | \
	XHYPASS_SELFTEST_TIMER_PPI | XHYPASS_SELFTEST_EVENT_CHANNEL | \
	XHYPASS_SELFTEST_TIMER_SPI)

enum xhypass_guest_mode {
	XHYPASS_GUEST_DYN = 0,
	XHYPASS_GUEST_RTO = 1,
};

struct xhypass_guest_info {
	__u32 abi_version;
	__u32 mode;
	__u32 rto_cpu;
	__u32 event_sgi;
};

struct xhypass_guest_stats {
	__u64 enter_attempts;
	__u64 enter_successes;
	__u64 exit_attempts;
	__u64 exit_successes;
	__u64 busy_returns;
	__u64 again_returns;
	__u64 other_errors;
};

struct xhypass_selftest_config {
	__u32 producer_cpu;
	__u32 timer_delay_us;
	__u32 event_timeout_ms;
	__u32 source_mask;
	__u32 spi_delay_us;
	__u32 reserved[3];
};

struct xhypass_notification_stats {
	__u64 produced;
	__u64 consumed;
	__u64 handler_entries;
	__u64 timeouts;
	__u64 unexpected;
	__u64 reordered;
	__u64 wrong_cpu;
};

struct xhypass_selftest_stats {
	__u32 running;
	__u32 producer_cpu;
	__u32 source_mask;
	__u32 reserved;
	__u64 iterations;
	struct xhypass_notification_stats ipi;
	struct xhypass_notification_stats timer;
	struct xhypass_notification_stats event_channel;
	struct xhypass_notification_stats spi;
	__u64 setup_errors;
	__u32 spi_linux_irq;
	__u32 spi_hwirq;
	__u32 spi_affinity_cpu;
	__u32 spi_timer_rate_hz;
	__u32 spi_affinity_verified;
	__u32 reserved2[3];
};

#define XHYPASS_IOC_MAGIC       'X'
#define XHYPASS_IOC_ENTER_RTO   _IO(XHYPASS_IOC_MAGIC, 0x01)
#define XHYPASS_IOC_EXIT_RTO    _IO(XHYPASS_IOC_MAGIC, 0x02)
#define XHYPASS_IOC_GET_INFO    _IOR(XHYPASS_IOC_MAGIC, 0x03, \
				     struct xhypass_guest_info)
#define XHYPASS_IOC_GET_STATS   _IOR(XHYPASS_IOC_MAGIC, 0x04, \
				     struct xhypass_guest_stats)
#define XHYPASS_IOC_RESET_STATS _IO(XHYPASS_IOC_MAGIC, 0x05)
#define XHYPASS_IOC_START_SELFTEST _IOW(XHYPASS_IOC_MAGIC, 0x06, \
				       struct xhypass_selftest_config)
#define XHYPASS_IOC_STOP_SELFTEST _IO(XHYPASS_IOC_MAGIC, 0x07)
#define XHYPASS_IOC_GET_SELFTEST_STATS _IOR(XHYPASS_IOC_MAGIC, 0x08, \
					    struct xhypass_selftest_stats)

#endif /* XHYPASS_IOCTL_H */
