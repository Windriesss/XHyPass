// SPDX-License-Identifier: GPL-2.0

#include <linux/delay.h>
#include <linux/clk.h>
#include <linux/completion.h>
#include <linux/errno.h>
#include <linux/hrtimer.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_irq.h>
#include <linux/preempt.h>
#include <linux/sched.h>
#include <linux/smp.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/wait.h>

#include <xen/events.h>
#include <xen/interface/event_channel.h>
#include <asm/xen/hypercall.h>

#include "xhypass_ioctl.h"

#define XHYPASS_HVC_IMM          0x4a48
#define XHYPASS_HVC_ENTER_RTO    12
#define XHYPASS_HVC_EXIT_RTO     13

#define XHYPASS_MAX_SGI          15

#define RK_TIMER_LOAD_COUNT0     0x00
#define RK_TIMER_LOAD_COUNT1     0x04
#define RK_TIMER_CONTROL         0x10
#define RK_TIMER_INT_STATUS      0x18
#define RK_TIMER_ENABLE          BIT(0)
#define RK_TIMER_USER_DEFINED    BIT(1)
#define RK_TIMER_INT_UNMASK      BIT(2)

/*
 * The selected vCPU must already be pinned one-to-one to its paired pCPU,
 * which must execute no other vCPU while RTO is active.
 */
static unsigned int rto_cpu = 6;
module_param(rto_cpu, uint, 0444);
MODULE_PARM_DESC(rto_cpu, "Linux CPU/vCPU used for RTO");

/*
 * Physical SGI used for event-channel wakeup notification. The platform must
 * reserve this SGI so that it is not used by another Xen IPI.
 */
static unsigned int event_sgi = 7;
module_param(event_sgi, uint, 0444);
MODULE_PARM_DESC(event_sgi, "Reserved physical SGI used for RTO event wakeup");

/*
 * module_exit() cannot return an error to rmmod. Retry transient RTO exit
 * failures for a bounded interval before reporting that Xen may remain in RTO.
 */
static unsigned int exit_retry_count = 1000;
module_param(exit_retry_count, uint, 0644);
MODULE_PARM_DESC(exit_retry_count, "Maximum number of RTO exit retries");

static unsigned int exit_retry_delay_ms = 1;
module_param(exit_retry_delay_ms, uint, 0644);
MODULE_PARM_DESC(exit_retry_delay_ms, "Delay between RTO exit retries");

/* Preserve the original load-to-enter behavior for existing experiments. */
static bool auto_enter = true;
module_param(auto_enter, bool, 0444);
MODULE_PARM_DESC(auto_enter, "Enter RTO while loading the module");

static char *spi_timer_path = "/timer@feae0000";
module_param(spi_timer_path, charp, 0444);
MODULE_PARM_DESC(spi_timer_path, "Device-tree path of the RK3588 SPI timer");

static unsigned int spi_expected_hwirq = 321;
module_param(spi_expected_hwirq, uint, 0444);
MODULE_PARM_DESC(spi_expected_hwirq, "Expected GIC INTID for the SPI timer");

static bool rto_enabled;
static DEFINE_MUTEX(control_lock);
static struct xhypass_guest_stats guest_stats;

struct xhypass_selftest_state {
	struct task_struct *producer;
	struct hrtimer timer;
	struct completion ipi_done;
	struct completion timer_done;
	struct completion spi_done;
	wait_queue_head_t event_wait;
	call_single_data_t ipi_csd;
	call_single_data_t timer_csd;
	evtchn_port_t event_rx_port;
	evtchn_port_t event_tx_port;
	int event_irq;
	struct device_node *spi_node;
	void __iomem *spi_base;
	struct clk *spi_pclk;
	struct clk *spi_clk;
	int spi_irq;
	cpumask_t spi_original_affinity;
	unsigned int producer_cpu;
	unsigned int timer_delay_us;
	unsigned int spi_delay_us;
	unsigned int event_timeout_ms;
	unsigned int source_mask;
	unsigned int spi_hwirq;
	unsigned int spi_linux_irq;
	unsigned long spi_timer_rate_hz;
	atomic64_t iterations;
	atomic64_t ipi_produced;
	atomic64_t ipi_consumed;
	atomic64_t ipi_handler_entries;
	atomic64_t ipi_timeouts;
	atomic64_t timer_produced;
	atomic64_t timer_consumed;
	atomic64_t timer_handler_entries;
	atomic64_t timer_timeouts;
	atomic64_t event_produced;
	atomic64_t event_consumed;
	atomic64_t event_handler_entries;
	atomic64_t event_timeouts;
	atomic64_t spi_produced;
	atomic64_t spi_consumed;
	atomic64_t spi_handler_entries;
	atomic64_t spi_timeouts;
	atomic64_t spi_unexpected;
	atomic64_t spi_wrong_cpu;
	atomic64_t setup_errors;
	atomic_t spi_armed;
	bool timer_initialized;
	bool spi_irq_requested;
	bool spi_pclk_enabled;
	bool spi_clk_enabled;
	bool spi_affinity_verified;
	bool running;
};

static struct xhypass_selftest_state selftest = {
	.event_rx_port = 0,
	.event_tx_port = 0,
	.event_irq = -1,
	.spi_irq = -1,
};

static void xhypass_spi_disable(struct xhypass_selftest_state *state)
{
	if (state->spi_base)
		writel_relaxed(0, state->spi_base + RK_TIMER_CONTROL);
}

static void xhypass_spi_clear(struct xhypass_selftest_state *state)
{
	if (state->spi_base)
		writel_relaxed(1, state->spi_base + RK_TIMER_INT_STATUS);
}

static irqreturn_t xhypass_spi_handler(int irq, void *data)
{
	struct xhypass_selftest_state *state = data;

	xhypass_spi_disable(state);
	xhypass_spi_clear(state);
	atomic64_inc(&state->spi_handler_entries);
	if (raw_smp_processor_id() != rto_cpu)
		atomic64_inc(&state->spi_wrong_cpu);
	if (atomic_cmpxchg(&state->spi_armed, 1, 0) != 1) {
		atomic64_inc(&state->spi_unexpected);
		return IRQ_HANDLED;
	}
	atomic64_inc(&state->spi_consumed);
	complete(&state->spi_done);
	return IRQ_HANDLED;
}

static int xhypass_spi_arm(struct xhypass_selftest_state *state)
{
	u64 cycles;

	cycles = DIV_ROUND_UP_ULL((u64)state->spi_timer_rate_hz *
				  state->spi_delay_us, USEC_PER_SEC);
	if (!cycles || cycles > U32_MAX)
		return -ERANGE;

	xhypass_spi_disable(state);
	xhypass_spi_clear(state);
	writel_relaxed((u32)cycles, state->spi_base + RK_TIMER_LOAD_COUNT0);
	writel_relaxed(0, state->spi_base + RK_TIMER_LOAD_COUNT1);
	atomic_set(&state->spi_armed, 1);
	wmb();
	writel_relaxed(RK_TIMER_ENABLE | RK_TIMER_USER_DEFINED |
		       RK_TIMER_INT_UNMASK, state->spi_base + RK_TIMER_CONTROL);
	return 0;
}

static void xhypass_cleanup_spi(struct xhypass_selftest_state *state)
{
	xhypass_spi_disable(state);
	xhypass_spi_clear(state);
	atomic_set(&state->spi_armed, 0);
	if (state->spi_irq_requested) {
		synchronize_irq(state->spi_irq);
		(void)irq_set_affinity(state->spi_irq,
				       &state->spi_original_affinity);
		free_irq(state->spi_irq, state);
		state->spi_irq_requested = false;
	}
	if (state->spi_irq >= 0) {
		irq_dispose_mapping(state->spi_irq);
		state->spi_irq = -1;
	}
	if (state->spi_clk_enabled) {
		clk_disable_unprepare(state->spi_clk);
		state->spi_clk_enabled = false;
	}
	if (!IS_ERR_OR_NULL(state->spi_clk)) {
		clk_put(state->spi_clk);
		state->spi_clk = NULL;
	}
	if (state->spi_pclk_enabled) {
		clk_disable_unprepare(state->spi_pclk);
		state->spi_pclk_enabled = false;
	}
	if (!IS_ERR_OR_NULL(state->spi_pclk)) {
		clk_put(state->spi_pclk);
		state->spi_pclk = NULL;
	}
	if (state->spi_base) {
		iounmap(state->spi_base);
		state->spi_base = NULL;
	}
	of_node_put(state->spi_node);
	state->spi_node = NULL;
}

static int xhypass_setup_spi(struct xhypass_selftest_state *state)
{
	struct irq_data *irq_data;
	const struct cpumask *effective;
	int rc;

	state->spi_node = of_find_node_by_path(spi_timer_path);
	if (!state->spi_node) {
		pr_err("XHyPass: SPI timer node %s is absent from the dom0 DT\n",
		       spi_timer_path);
		return -ENODEV;
	}
	if (!of_device_is_available(state->spi_node)) {
		pr_err("XHyPass: SPI timer node %s is disabled in the dom0 DT\n",
		       spi_timer_path);
		return -EACCES;
	}
	if (!of_device_is_compatible(state->spi_node,
				     "xhypass,rk3588-spi-timer") &&
	    !of_device_is_compatible(state->spi_node,
				     "rockchip,rk3288-timer")) {
		pr_err("XHyPass: incompatible SPI timer node %s\n",
		       spi_timer_path);
		return -EINVAL;
	}
	state->spi_base = of_iomap(state->spi_node, 0);
	if (!state->spi_base)
		return -ENOMEM;

	state->spi_pclk = of_clk_get_by_name(state->spi_node, "pclk");
	if (IS_ERR(state->spi_pclk)) {
		rc = PTR_ERR(state->spi_pclk);
		state->spi_pclk = NULL;
		return rc;
	}
	rc = clk_prepare_enable(state->spi_pclk);
	if (rc)
		return rc;
	state->spi_pclk_enabled = true;

	state->spi_clk = of_clk_get_by_name(state->spi_node, "timer");
	if (IS_ERR(state->spi_clk)) {
		rc = PTR_ERR(state->spi_clk);
		state->spi_clk = NULL;
		return rc;
	}
	rc = clk_prepare_enable(state->spi_clk);
	if (rc)
		return rc;
	state->spi_clk_enabled = true;
	state->spi_timer_rate_hz = clk_get_rate(state->spi_clk);
	if (!state->spi_timer_rate_hz)
		return -EINVAL;

	state->spi_irq = irq_of_parse_and_map(state->spi_node, 0);
	if (!state->spi_irq)
		return -EINVAL;
	irq_data = irq_get_irq_data(state->spi_irq);
	if (!irq_data)
		return -EINVAL;
	state->spi_linux_irq = state->spi_irq;
	state->spi_hwirq = irqd_to_hwirq(irq_data);
	if (state->spi_hwirq != spi_expected_hwirq) {
		pr_err("XHyPass: SPI timer hwirq=%u, expected=%u\n",
		       state->spi_hwirq, spi_expected_hwirq);
		return -EINVAL;
	}

	xhypass_spi_disable(state);
	xhypass_spi_clear(state);
	rc = request_irq(state->spi_irq, xhypass_spi_handler, IRQF_TIMER,
			 "xhypass-spi-timer", state);
	if (rc) {
		pr_err("XHyPass: request_irq(%d/hwirq%u) failed: rc=%d\n",
		       state->spi_irq, state->spi_hwirq, rc);
		return rc;
	}
	state->spi_irq_requested = true;
	cpumask_copy(&state->spi_original_affinity,
		     irq_data_get_affinity_mask(irq_data));
	rc = irq_set_affinity(state->spi_irq, cpumask_of(rto_cpu));
	if (rc)
		return rc;
	effective = irq_data_get_effective_affinity_mask(irq_data);
	if (!effective || !cpumask_equal(effective, cpumask_of(rto_cpu))) {
		pr_err("XHyPass: SPI IRQ%d effective affinity is not CPU%u\n",
		       state->spi_irq, rto_cpu);
		return -EXDEV;
	}
	state->spi_affinity_verified = true;
	pr_info("XHyPass: SPI timer %s IRQ%d/hwirq%u bound to CPU%u at %lu Hz\n",
		spi_timer_path, state->spi_irq, state->spi_hwirq, rto_cpu,
		state->spi_timer_rate_hz);
	return 0;
}

static void xhypass_close_evtchn(evtchn_port_t port)
{
	struct evtchn_close close = { .port = port };

	if (port)
		(void)HYPERVISOR_event_channel_op(EVTCHNOP_close, &close);
}

static irqreturn_t xhypass_event_handler(int irq, void *data)
{
	struct xhypass_selftest_state *state = data;

	atomic64_inc(&state->event_handler_entries);
	atomic64_inc(&state->event_consumed);
	wake_up(&state->event_wait);
	return IRQ_HANDLED;
}

static void xhypass_ipi_handler(void *data)
{
	struct xhypass_selftest_state *state = data;

	atomic64_inc(&state->ipi_handler_entries);
	atomic64_inc(&state->ipi_consumed);
	/* IPIs remain hard IRQs on PREEMPT_RT; avoid waitqueue RT locks here. */
	complete(&state->ipi_done);
}

static enum hrtimer_restart xhypass_timer_handler(struct hrtimer *timer)
{
	struct xhypass_selftest_state *state =
		container_of(timer, struct xhypass_selftest_state, timer);

	atomic64_inc(&state->timer_handler_entries);
	atomic64_inc(&state->timer_consumed);
	complete(&state->timer_done);
	return HRTIMER_NORESTART;
}

static void xhypass_start_timer_on_target(void *data)
{
	struct xhypass_selftest_state *state = data;
	ktime_t delay = ns_to_ktime((u64)state->timer_delay_us * NSEC_PER_USEC);

	hrtimer_start(&state->timer, delay, HRTIMER_MODE_REL_PINNED);
}

static int xhypass_selftest_thread(void *data)
{
	struct xhypass_selftest_state *state = data;
	unsigned long timeout;
	u64 sequence;
	int rc;

	timeout = msecs_to_jiffies(state->event_timeout_ms);
	if (!timeout)
		timeout = 1;

	while (!kthread_should_stop()) {
		if (state->source_mask & XHYPASS_SELFTEST_SGI) {
			reinit_completion(&state->ipi_done);
			sequence = atomic64_inc_return(&state->ipi_produced);
			rc = smp_call_function_single_async(rto_cpu,
							&state->ipi_csd);
			if (rc || !wait_for_completion_timeout(&state->ipi_done,
							      timeout)) {
				atomic64_inc(&state->ipi_timeouts);
				goto wait_for_stop;
			}
			if (atomic64_read(&state->ipi_consumed) < sequence) {
				atomic64_inc(&state->ipi_timeouts);
				goto wait_for_stop;
			}
		}

		if (state->source_mask & XHYPASS_SELFTEST_TIMER_PPI) {
			reinit_completion(&state->timer_done);
			atomic64_inc(&state->timer_produced);
			rc = smp_call_function_single_async(rto_cpu,
							&state->timer_csd);
			if (rc || !wait_for_completion_timeout(&state->timer_done,
							      timeout)) {
				atomic64_inc(&state->timer_timeouts);
				goto wait_for_stop;
			}
		}

		if (state->source_mask & XHYPASS_SELFTEST_EVENT_CHANNEL) {
			sequence = atomic64_inc_return(&state->event_produced);
			notify_remote_via_evtchn(state->event_tx_port);
			if (!wait_event_timeout(state->event_wait,
					atomic64_read(&state->event_consumed) >= sequence ||
					kthread_should_stop(), timeout)) {
				atomic64_inc(&state->event_timeouts);
				goto wait_for_stop;
			}
		}

		if (state->source_mask & XHYPASS_SELFTEST_TIMER_SPI) {
			reinit_completion(&state->spi_done);
			sequence = atomic64_inc_return(&state->spi_produced);
			rc = xhypass_spi_arm(state);
			if (rc || !wait_for_completion_timeout(&state->spi_done,
							      timeout)) {
				atomic_set(&state->spi_armed, 0);
				atomic64_inc(&state->spi_timeouts);
				goto wait_for_stop;
			}
			if (atomic64_read(&state->spi_consumed) < sequence) {
				atomic64_inc(&state->spi_timeouts);
				goto wait_for_stop;
			}
		}

		atomic64_inc(&state->iterations);
		cond_resched();
	}

	return 0;

wait_for_stop:
	/*
	 * Keep the task alive after a failed source.  The controller still needs
	 * to collect counters and call kthread_stop(); returning here would leave
	 * selftest.producer pointing at a task whose final reference can vanish.
	 */
	while (!kthread_should_stop())
		schedule_timeout_interruptible(MAX_SCHEDULE_TIMEOUT);
	return 0;
}

static void xhypass_reset_selftest_stats(void)
{
	atomic64_set(&selftest.iterations, 0);
	atomic64_set(&selftest.ipi_produced, 0);
	atomic64_set(&selftest.ipi_consumed, 0);
	atomic64_set(&selftest.ipi_handler_entries, 0);
	atomic64_set(&selftest.ipi_timeouts, 0);
	atomic64_set(&selftest.timer_produced, 0);
	atomic64_set(&selftest.timer_consumed, 0);
	atomic64_set(&selftest.timer_handler_entries, 0);
	atomic64_set(&selftest.timer_timeouts, 0);
	atomic64_set(&selftest.event_produced, 0);
	atomic64_set(&selftest.event_consumed, 0);
	atomic64_set(&selftest.event_handler_entries, 0);
	atomic64_set(&selftest.event_timeouts, 0);
	atomic64_set(&selftest.spi_produced, 0);
	atomic64_set(&selftest.spi_consumed, 0);
	atomic64_set(&selftest.spi_handler_entries, 0);
	atomic64_set(&selftest.spi_timeouts, 0);
	atomic64_set(&selftest.spi_unexpected, 0);
	atomic64_set(&selftest.spi_wrong_cpu, 0);
	atomic64_set(&selftest.setup_errors, 0);
	atomic_set(&selftest.spi_armed, 0);
	selftest.spi_hwirq = 0;
	selftest.spi_linux_irq = 0;
	selftest.spi_timer_rate_hz = 0;
	selftest.spi_affinity_verified = false;
}

static void xhypass_stop_selftest(void)
{
	if (selftest.producer) {
		kthread_stop(selftest.producer);
		selftest.producer = NULL;
	}
	if (selftest.timer_initialized) {
		hrtimer_cancel(&selftest.timer);
		selftest.timer_initialized = false;
	}
	if (selftest.event_irq >= 0) {
		unbind_from_irqhandler(selftest.event_irq, &selftest);
		selftest.event_irq = -1;
		selftest.event_rx_port = 0;
	}
	xhypass_close_evtchn(selftest.event_rx_port);
	selftest.event_rx_port = 0;
	xhypass_close_evtchn(selftest.event_tx_port);
	selftest.event_tx_port = 0;
	xhypass_cleanup_spi(&selftest);
	selftest.running = false;
}

static int xhypass_start_selftest(const struct xhypass_selftest_config *config)
{
	struct evtchn_alloc_unbound alloc = {
		.dom = DOMID_SELF,
		.remote_dom = DOMID_SELF,
	};
	struct evtchn_bind_interdomain bind = { .remote_dom = DOMID_SELF };
	int rc;

	if (selftest.running)
		return -EBUSY;
	if (config->producer_cpu >= nr_cpu_ids ||
	    !cpu_online(config->producer_cpu) ||
	    config->producer_cpu == rto_cpu ||
	    !cpu_online(rto_cpu) ||
	    !config->timer_delay_us || !config->event_timeout_ms ||
	    ((config->source_mask & XHYPASS_SELFTEST_TIMER_SPI) &&
	     !config->spi_delay_us) ||
	    !config->source_mask ||
	    (config->source_mask & ~XHYPASS_SELFTEST_ALL))
		return -EINVAL;

	xhypass_reset_selftest_stats();
	selftest.producer_cpu = config->producer_cpu;
	selftest.timer_delay_us = config->timer_delay_us;
	selftest.spi_delay_us = config->spi_delay_us;
	selftest.event_timeout_ms = config->event_timeout_ms;
	selftest.source_mask = config->source_mask;
	init_completion(&selftest.ipi_done);
	init_completion(&selftest.timer_done);
	init_completion(&selftest.spi_done);
	init_waitqueue_head(&selftest.event_wait);
	memset(&selftest.ipi_csd, 0, sizeof(selftest.ipi_csd));
	selftest.ipi_csd.func = xhypass_ipi_handler;
	selftest.ipi_csd.info = &selftest;
	memset(&selftest.timer_csd, 0, sizeof(selftest.timer_csd));
	selftest.timer_csd.func = xhypass_start_timer_on_target;
	selftest.timer_csd.info = &selftest;
	if (selftest.source_mask & XHYPASS_SELFTEST_TIMER_PPI) {
		hrtimer_init(&selftest.timer, CLOCK_MONOTONIC,
			     HRTIMER_MODE_REL_PINNED);
		selftest.timer.function = xhypass_timer_handler;
		selftest.timer_initialized = true;
	}

	if (selftest.source_mask & XHYPASS_SELFTEST_EVENT_CHANNEL) {
		rc = HYPERVISOR_event_channel_op(EVTCHNOP_alloc_unbound, &alloc);
		if (rc)
			goto fail;
		selftest.event_rx_port = alloc.port;
		bind.remote_port = alloc.port;
		rc = HYPERVISOR_event_channel_op(EVTCHNOP_bind_interdomain, &bind);
		if (rc)
			goto fail;
		selftest.event_tx_port = bind.local_port;

		rc = bind_evtchn_to_irqhandler(selftest.event_rx_port,
						xhypass_event_handler, 0,
						"xhypass-selftest", &selftest);
		if (rc < 0)
			goto fail;
		selftest.event_irq = rc;
		rc = irq_set_affinity(selftest.event_irq, cpumask_of(rto_cpu));
		if (rc)
			goto fail;
	}

	if (selftest.source_mask & XHYPASS_SELFTEST_TIMER_SPI) {
		rc = xhypass_setup_spi(&selftest);
		if (rc)
			goto fail;
	}

	selftest.producer = kthread_create(xhypass_selftest_thread, &selftest,
					   "xhypass-producer");
	if (IS_ERR(selftest.producer)) {
		rc = PTR_ERR(selftest.producer);
		selftest.producer = NULL;
		goto fail;
	}
	kthread_bind(selftest.producer, selftest.producer_cpu);
	selftest.running = true;
	wake_up_process(selftest.producer);
	return 0;

fail:
	atomic64_inc(&selftest.setup_errors);
	xhypass_stop_selftest();
	return rc;
}

static void xhypass_copy_notification_stats(
	struct xhypass_notification_stats *destination,
	atomic64_t *produced, atomic64_t *consumed,
	atomic64_t *handler_entries, atomic64_t *timeouts)
{
	destination->produced = atomic64_read(produced);
	destination->consumed = atomic64_read(consumed);
	destination->handler_entries = atomic64_read(handler_entries);
	destination->timeouts = atomic64_read(timeouts);
}

static void xhypass_get_selftest_stats(struct xhypass_selftest_stats *stats)
{
	memset(stats, 0, sizeof(*stats));
	stats->running = selftest.running;
	stats->producer_cpu = selftest.producer_cpu;
	stats->source_mask = selftest.source_mask;
	stats->iterations = atomic64_read(&selftest.iterations);
	xhypass_copy_notification_stats(&stats->ipi, &selftest.ipi_produced,
		&selftest.ipi_consumed, &selftest.ipi_handler_entries,
		&selftest.ipi_timeouts);
	xhypass_copy_notification_stats(&stats->timer, &selftest.timer_produced,
		&selftest.timer_consumed, &selftest.timer_handler_entries,
		&selftest.timer_timeouts);
	xhypass_copy_notification_stats(&stats->event_channel,
		&selftest.event_produced, &selftest.event_consumed,
		&selftest.event_handler_entries, &selftest.event_timeouts);
	xhypass_copy_notification_stats(&stats->spi, &selftest.spi_produced,
		&selftest.spi_consumed, &selftest.spi_handler_entries,
		&selftest.spi_timeouts);
	stats->spi.unexpected = atomic64_read(&selftest.spi_unexpected);
	stats->spi.wrong_cpu = atomic64_read(&selftest.spi_wrong_cpu);
	stats->setup_errors = atomic64_read(&selftest.setup_errors);
	stats->spi_linux_irq = selftest.spi_linux_irq;
	stats->spi_hwirq = selftest.spi_hwirq;
	stats->spi_affinity_cpu = rto_cpu;
	stats->spi_timer_rate_hz = selftest.spi_timer_rate_hz;
	stats->spi_affinity_verified = selftest.spi_affinity_verified;
}

/*
 * XHyPass private HVC ABI:
 *
 * x0: operation number on entry, return value on exit
 * x1: operation argument
 */
static __always_inline long xhypass_hvc(unsigned long operation,
					unsigned long argument)
{
	register unsigned long x0 asm("x0") = operation;
	register unsigned long x1 asm("x1") = argument;

	asm volatile(
		"hvc #0x4a48"
		: "+r" (x0), "+r" (x1)
		:
		: "memory");

	return (long)x0;
}

/*
 * The HVC must be issued from ordinary process context on the target vCPU.
 * Disable preemption between the CPU check and the caller-local HVC. Xen masks
 * EL2 local interrupts in its HVC handler, so EL1 IRQs remain enabled here.
 */
static long xhypass_switch_mode(bool enter)
{
	unsigned int current_cpu;
	long rc;

	if (WARN_ON_ONCE(in_interrupt())) {
		pr_err("XHyPass: HVC called from interrupt context\n");
		return -EBUSY;
	}

	if (WARN_ON_ONCE(irqs_disabled())) {
		pr_err("XHyPass: HVC unexpectedly called with EL1 IRQs disabled\n");
		return -EBUSY;
	}

	preempt_disable();
	current_cpu = smp_processor_id();

	if (current_cpu != rto_cpu) {
		preempt_enable();
		pr_err("XHyPass: running on CPU%u, expected CPU%u\n",
		       current_cpu, rto_cpu);
		return -EXDEV;
	}

	if (enter)
		rc = xhypass_hvc(XHYPASS_HVC_ENTER_RTO, event_sgi);
	else
		rc = xhypass_hvc(XHYPASS_HVC_EXIT_RTO, 0);

	preempt_enable();

	return rc;
}

static void xhypass_account_result(bool enter, long rc)
{
	if (enter)
		guest_stats.enter_attempts++;
	else
		guest_stats.exit_attempts++;

	if (!rc) {
		if (enter)
			guest_stats.enter_successes++;
		else
			guest_stats.exit_successes++;
	} else if (rc == -EBUSY) {
		guest_stats.busy_returns++;
	} else if (rc == -EAGAIN) {
		guest_stats.again_returns++;
	} else {
		guest_stats.other_errors++;
	}
}

static long xhypass_request_mode(bool enter)
{
	long rc;

	if (enter == READ_ONCE(rto_enabled))
		return -EALREADY;

	rc = xhypass_switch_mode(enter);
	xhypass_account_result(enter, rc);
	if (!rc)
		WRITE_ONCE(rto_enabled, enter);

	return rc;
}

static long xhypass_ioctl(struct file *file, unsigned int command,
			  unsigned long argument)
{
	void __user *user = (void __user *)argument;
	struct xhypass_guest_info info;
	struct xhypass_selftest_config selftest_config;
	struct xhypass_selftest_stats selftest_stats;
	long rc = 0;

	if (_IOC_TYPE(command) != XHYPASS_IOC_MAGIC)
		return -ENOTTY;

	mutex_lock(&control_lock);
	switch (command) {
	case XHYPASS_IOC_ENTER_RTO:
		rc = xhypass_request_mode(true);
		break;
	case XHYPASS_IOC_EXIT_RTO:
		rc = xhypass_request_mode(false);
		break;
	case XHYPASS_IOC_GET_INFO:
		info.abi_version = XHYPASS_IOCTL_ABI_VERSION;
		info.mode = rto_enabled ? XHYPASS_GUEST_RTO : XHYPASS_GUEST_DYN;
		info.rto_cpu = rto_cpu;
		info.event_sgi = event_sgi;
		if (copy_to_user(user, &info, sizeof(info)))
			rc = -EFAULT;
		break;
	case XHYPASS_IOC_GET_STATS:
		if (copy_to_user(user, &guest_stats, sizeof(guest_stats)))
			rc = -EFAULT;
		break;
	case XHYPASS_IOC_RESET_STATS:
		memset(&guest_stats, 0, sizeof(guest_stats));
		break;
	case XHYPASS_IOC_START_SELFTEST:
		if (copy_from_user(&selftest_config, user,
				   sizeof(selftest_config))) {
			rc = -EFAULT;
			break;
		}
		rc = xhypass_start_selftest(&selftest_config);
		break;
	case XHYPASS_IOC_STOP_SELFTEST:
		xhypass_stop_selftest();
		break;
	case XHYPASS_IOC_GET_SELFTEST_STATS:
		xhypass_get_selftest_stats(&selftest_stats);
		if (copy_to_user(user, &selftest_stats,
				 sizeof(selftest_stats)))
			rc = -EFAULT;
		break;
	default:
		rc = -ENOTTY;
		break;
	}
	mutex_unlock(&control_lock);

	return rc;
}

static const struct file_operations xhypass_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = xhypass_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = xhypass_ioctl,
#endif
};

static struct miscdevice xhypass_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "xhypass",
	.fops = &xhypass_fops,
	.mode = 0600,
};

static int __init interrupt_passthrough_init(void)
{
	long rc;

	if (event_sgi > XHYPASS_MAX_SGI) {
		pr_err("XHyPass: invalid event SGI %u; valid SGIs are 0-%u, "
		       "and Xen must approve the selected non-static SGI\n",
		       event_sgi, XHYPASS_MAX_SGI);
		return -EINVAL;
	}

	rc = misc_register(&xhypass_miscdev);
	if (rc) {
		pr_err("XHyPass: failed to register /dev/xhypass: rc=%ld\n", rc);
		return (int)rc;
	}

	if (auto_enter) {
		pr_info("XHyPass: requesting DYN->RTO: CPU%u, event SGI%u\n",
			rto_cpu, event_sgi);
		mutex_lock(&control_lock);
		rc = xhypass_request_mode(true);
		mutex_unlock(&control_lock);
		if (rc) {
			pr_err("XHyPass: DYN->RTO failed: rc=%ld\n", rc);
			misc_deregister(&xhypass_miscdev);
			return (int)rc;
		}
		pr_info("XHyPass: DYN->RTO completed on CPU%u\n", rto_cpu);
	}

	pr_info("XHyPass: control device ready at /dev/xhypass\n");

	return 0;
}

static void __exit interrupt_passthrough_exit(void)
{
	unsigned int attempt;
	long rc = 0;

	mutex_lock(&control_lock);
	xhypass_stop_selftest();
	misc_deregister(&xhypass_miscdev);
	if (!READ_ONCE(rto_enabled)) {
		pr_info("XHyPass: RTO was not enabled\n");
		mutex_unlock(&control_lock);
		return;
	}

	pr_info("XHyPass: requesting RTO->DYN on CPU%u\n", rto_cpu);

	for (attempt = 0; attempt <= exit_retry_count; attempt++) {
		rc = xhypass_request_mode(false);

		if (!rc) {
			pr_info("XHyPass: RTO->DYN completed on CPU%u "
				"after %u retries\n", rto_cpu, attempt);
			mutex_unlock(&control_lock);
			return;
		}

		/*
		 * -EBUSY covers Active interrupt ownership or an unexpected mode.
		 * -EAGAIN covers transient Active-and-Pending or late IRQ windows.
		 */
		if (rc != -EBUSY && rc != -EAGAIN)
			break;

		if (attempt == exit_retry_count)
			break;

		if (attempt == 0 || !(attempt % 100))
			pr_warn("XHyPass: RTO->DYN retry %u/%u, rc=%ld\n",
				attempt + 1, exit_retry_count, rc);

		/* Wait for a guest ISR, remote Xen IRQ handler, or DIR to finish. */
		if (exit_retry_delay_ms)
			msleep(exit_retry_delay_ms);
		else
			cond_resched();
	}

	/*
	 * The module will be unloaded even though Xen may still be in RTO because
	 * module_exit() cannot reject rmmod.
	 */
	pr_emerg("XHyPass: RTO->DYN failed permanently: rc=%ld, "
		 "attempts=%u; Xen may remain in RTO mode\n",
		 rc, attempt + 1);
	mutex_unlock(&control_lock);
}

module_init(interrupt_passthrough_init);
module_exit(interrupt_passthrough_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("XHyPass");
MODULE_DESCRIPTION("XHyPass RTO interrupt ownership switch");
MODULE_VERSION("1.0");
