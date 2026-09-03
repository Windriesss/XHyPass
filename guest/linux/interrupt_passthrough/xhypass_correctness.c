// SPDX-License-Identifier: GPL-2.0

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "xhypass_ioctl.h"

struct options {
	const char *device;
	const char *output;
	const char *platform;
	const char *run_id;
	const char *condition;
	unsigned long iterations;
	unsigned int producer_cpu;
	unsigned int timer_delay_us;
	unsigned int spi_delay_us;
	unsigned int event_timeout_ms;
	unsigned int max_retries;
	unsigned int retry_delay_us;
	unsigned int dwell_us;
	unsigned int source_mask;
};

static void usage(const char *program)
{
	fprintf(stderr,
		"Usage: %s [options]\n"
		"  --iterations N          DYN/RTO pairs (default 10000)\n"
		"  --producer-cpu N        non-RTO producer CPU (default 0)\n"
		"  --timer-delay-us N      local timer delay (default 50)\n"
		"  --spi-delay-us N        RK3588 SPI timer delay (default 50)\n"
		"  --event-timeout-ms N    watchdog (default 1000)\n"
		"  --max-retries N         retries per transition (default 1000)\n"
		"  --retry-delay-us N      retry interval (default 10)\n"
		"  --dwell-us N            time in each stable mode (default 0)\n"
		"  --source NAME           sgi, timer, event, spi, or all (default all)\n"
		"  --platform NAME         label (default RK3588)\n"
		"  --run-id ID             identifier (default run-001)\n"
		"  --condition NAME        label (default notification-stress)\n"
		"  --output PATH           output JSON (required)\n"
		"  --device PATH           control device (default /dev/xhypass)\n",
		program);
}

static int parse_source(const char *text, unsigned int *mask)
{
	if (!strcmp(text, "sgi"))
		*mask = XHYPASS_SELFTEST_SGI;
	else if (!strcmp(text, "timer"))
		*mask = XHYPASS_SELFTEST_TIMER_PPI;
	else if (!strcmp(text, "event"))
		*mask = XHYPASS_SELFTEST_EVENT_CHANNEL;
	else if (!strcmp(text, "spi"))
		*mask = XHYPASS_SELFTEST_TIMER_SPI;
	else if (!strcmp(text, "all"))
		*mask = XHYPASS_SELFTEST_ALL;
	else
		return -1;
	return 0;
}

static int parse_unsigned(const char *text, unsigned long *value)
{
	char *end;
	unsigned long parsed;

	errno = 0;
	parsed = strtoul(text, &end, 10);
	if (errno || !text[0] || *end)
		return -1;
	*value = parsed;
	return 0;
}

static bool safe_label(const char *text)
{
	const unsigned char *cursor = (const unsigned char *)text;

	if (!*cursor)
		return false;
	for (; *cursor; cursor++)
		if (!((*cursor >= 'a' && *cursor <= 'z') ||
		      (*cursor >= 'A' && *cursor <= 'Z') ||
		      (*cursor >= '0' && *cursor <= '9') ||
		      *cursor == '_' || *cursor == '-' || *cursor == '.'))
			return false;
	return true;
}

static int request_mode(int fd, unsigned long request,
			unsigned int max_retries, unsigned int retry_delay_us,
			unsigned int *retry_count)
{
	unsigned int attempt;

	for (attempt = 0; attempt <= max_retries; attempt++) {
		if (!ioctl(fd, request)) {
			*retry_count += attempt;
			return 0;
		}
		if (errno != EBUSY && errno != EAGAIN)
			return -errno;
		if (attempt == max_retries)
			return -errno;
		if (retry_delay_us)
			usleep(retry_delay_us);
	}
	return -EIO;
}

static uint64_t lost_count(const struct xhypass_notification_stats *stats)
{
	return stats->produced > stats->consumed ?
		stats->produced - stats->consumed : 0;
}

static uint64_t duplicate_count(const struct xhypass_notification_stats *stats)
{
	return stats->consumed > stats->produced ?
		stats->consumed - stats->produced : 0;
}

static void write_notification(FILE *output, const char *name,
			       const struct xhypass_notification_stats *stats,
			       bool enabled, bool comma)
{
	fprintf(output,
		"    \"%s\": {\"skipped\": %s, \"produced\": %llu, "
		"\"consumed\": %llu, "
		"\"handler_entries\": %llu, \"lost\": %llu, "
		"\"duplicates\": %llu, \"unexpected\": %llu, "
		"\"reordered\": %llu, \"wrong_cpu\": %llu, "
		"\"timeouts\": %llu}%s\n",
		name, enabled ? "false" : "true",
		(unsigned long long)stats->produced,
		(unsigned long long)stats->consumed,
		(unsigned long long)stats->handler_entries,
		(unsigned long long)lost_count(stats),
		(unsigned long long)duplicate_count(stats),
		(unsigned long long)stats->unexpected,
		(unsigned long long)stats->reordered,
		(unsigned long long)stats->wrong_cpu,
		(unsigned long long)stats->timeouts, comma ? "," : "");
}

static bool exact_notification(const struct xhypass_notification_stats *stats)
{
	return !lost_count(stats) && !duplicate_count(stats) &&
		!stats->unexpected && !stats->reordered && !stats->wrong_cpu;
}

static int write_result(const struct options *options,
			const struct xhypass_guest_info *info,
			const struct xhypass_guest_stats *guest,
			const struct xhypass_selftest_stats *selftest,
			unsigned long completed_pairs,
			unsigned int enter_retries, unsigned int exit_retries,
			int transition_rc)
{
	uint64_t watchdogs = selftest->ipi.timeouts + selftest->timer.timeouts +
		selftest->event_channel.timeouts + selftest->spi.timeouts;
	bool exact = (!(options->source_mask & XHYPASS_SELFTEST_SGI) ||
		exact_notification(&selftest->ipi)) &&
		(!(options->source_mask & XHYPASS_SELFTEST_TIMER_PPI) ||
		 exact_notification(&selftest->timer)) &&
		(!(options->source_mask & XHYPASS_SELFTEST_EVENT_CHANNEL) ||
		 exact_notification(&selftest->event_channel)) &&
		(!(options->source_mask & XHYPASS_SELFTEST_TIMER_SPI) ||
		 (exact_notification(&selftest->spi) &&
		  selftest->spi_affinity_verified));
	bool passed = transition_rc == 0 && info->mode == XHYPASS_GUEST_DYN &&
		watchdogs == 0 && exact && selftest->setup_errors == 0 &&
		completed_pairs == options->iterations;
	FILE *output = fopen(options->output, "w");

	if (!output) {
		perror(options->output);
		return -errno;
	}
	fprintf(output,
		"{\n"
		"  \"schema_version\": 1,\n"
		"  \"platform\": \"%s\",\n"
		"  \"run_id\": \"%s\",\n"
		"  \"condition\": \"%s\",\n"
		"  \"scope\": \"Xen-dom0-Linux\",\n"
		"  \"source_mask\": %u,\n"
		"  \"transition_pairs\": %lu,\n"
		"  \"notification_iterations\": %llu,\n"
		"  \"watchdog_timeouts\": %llu,\n"
		"  \"final_mode\": \"%s\",\n"
		"  \"passed\": %s,\n"
		"  \"transition_counters\": {\"enter_attempts\": %llu, "
		"\"enter_successes\": %llu, \"exit_attempts\": %llu, "
		"\"exit_successes\": %llu, \"busy_returns\": %llu, "
		"\"again_returns\": %llu, \"other_errors\": %llu, "
		"\"enter_retries\": %u, \"exit_retries\": %u},\n"
		"  \"notifications\": {\n",
		options->platform, options->run_id, options->condition,
		options->source_mask,
		completed_pairs, (unsigned long long)selftest->iterations,
		(unsigned long long)watchdogs,
		info->mode == XHYPASS_GUEST_DYN ? "DYN" : "RTO",
		passed ? "true" : "false",
		(unsigned long long)guest->enter_attempts,
		(unsigned long long)guest->enter_successes,
		(unsigned long long)guest->exit_attempts,
		(unsigned long long)guest->exit_successes,
		(unsigned long long)guest->busy_returns,
		(unsigned long long)guest->again_returns,
		(unsigned long long)guest->other_errors,
		enter_retries, exit_retries);
	write_notification(output, "SGI", &selftest->ipi,
		options->source_mask & XHYPASS_SELFTEST_SGI, true);
	write_notification(output, "timer-PPI", &selftest->timer,
		options->source_mask & XHYPASS_SELFTEST_TIMER_PPI, true);
	write_notification(output, "event-channel", &selftest->event_channel,
		options->source_mask & XHYPASS_SELFTEST_EVENT_CHANNEL, true);
	write_notification(output, "device-SPI", &selftest->spi,
		options->source_mask & XHYPASS_SELFTEST_TIMER_SPI, false);
	fprintf(output,
		"  },\n"
		"  \"spi_timer\": {\"device_tree_path\": \"/timer@feae0000\", "
		"\"linux_irq\": %u, \"gic_hwirq\": %u, \"affinity_cpu\": %u, "
		"\"timer_rate_hz\": %u, \"affinity_verified\": %s},\n"
		"  \"state_cases\": [\n"
		"    {\"name\": \"DYN-to-RTO-under-notification-load\", "
		"\"expected_rc\": 0, \"observed_rc\": %d, "
		"\"final_mode\": \"RTO\", \"passed\": %s},\n"
		"    {\"name\": \"RTO-to-DYN-under-notification-load\", "
		"\"expected_rc\": 0, \"observed_rc\": %d, "
		"\"final_mode\": \"%s\", \"passed\": %s}\n"
		"  ],\n"
		"  \"setup_errors\": %llu\n"
		"}\n",
		selftest->spi_linux_irq, selftest->spi_hwirq,
		selftest->spi_affinity_cpu, selftest->spi_timer_rate_hz,
		selftest->spi_affinity_verified ? "true" : "false",
		transition_rc, transition_rc ? "false" : "true",
		transition_rc, info->mode == XHYPASS_GUEST_DYN ? "DYN" : "RTO",
		passed ? "true" : "false",
		(unsigned long long)selftest->setup_errors);
	if (fclose(output)) {
		perror(options->output);
		return -errno;
	}
	return passed ? 0 : -EIO;
}

int main(int argc, char **argv)
{
	struct options options = {
		.device = XHYPASS_DEVICE_PATH, .platform = "RK3588",
		.run_id = "run-001", .condition = "notification-stress",
		.iterations = 10000, .producer_cpu = 0,
		.timer_delay_us = 50, .event_timeout_ms = 1000,
		.spi_delay_us = 50,
		.max_retries = 1000, .retry_delay_us = 10,
		.source_mask = XHYPASS_SELFTEST_ALL,
	};
	static const struct option long_options[] = {
		{ "iterations", required_argument, NULL, 'n' },
		{ "producer-cpu", required_argument, NULL, 'p' },
		{ "timer-delay-us", required_argument, NULL, 't' },
		{ "spi-delay-us", required_argument, NULL, 'S' },
		{ "event-timeout-ms", required_argument, NULL, 'w' },
		{ "max-retries", required_argument, NULL, 'm' },
		{ "retry-delay-us", required_argument, NULL, 'r' },
		{ "dwell-us", required_argument, NULL, 'd' },
		{ "source", required_argument, NULL, 's' },
		{ "platform", required_argument, NULL, 'P' },
		{ "run-id", required_argument, NULL, 'i' },
		{ "condition", required_argument, NULL, 'c' },
		{ "output", required_argument, NULL, 'o' },
		{ "device", required_argument, NULL, 'D' },
		{ "help", no_argument, NULL, 'h' }, { NULL, 0, NULL, 0 },
	};
	struct xhypass_selftest_config config;
	struct xhypass_selftest_stats selftest_stats = {0};
	struct xhypass_guest_stats guest_stats = {0};
	struct xhypass_guest_info info = {0};
	unsigned int enter_retries = 0, exit_retries = 0;
	unsigned long value, completed = 0;
	int fd = -1, option, rc = 0, result;
	bool selftest_started = false;

	while ((option = getopt_long(argc, argv, "n:p:t:S:w:m:r:d:s:P:i:c:o:D:h",
				     long_options, NULL)) != -1) {
		if (option == 'h') { usage(argv[0]); return 0; }
		if (option == 's') {
			if (parse_source(optarg, &options.source_mask)) {
				fprintf(stderr, "invalid source: %s\n", optarg);
				return 2;
			}
			continue;
		}
		if (strchr("nptSwmrd", option)) {
			if (parse_unsigned(optarg, &value)) {
				fprintf(stderr, "invalid numeric value: %s\n", optarg);
				return 2;
			}
			switch (option) {
			case 'n': options.iterations = value; break;
			case 'p': options.producer_cpu = value; break;
			case 't': options.timer_delay_us = value; break;
			case 'S': options.spi_delay_us = value; break;
			case 'w': options.event_timeout_ms = value; break;
			case 'm': options.max_retries = value; break;
			case 'r': options.retry_delay_us = value; break;
			case 'd': options.dwell_us = value; break;
			}
			continue;
		}
		switch (option) {
		case 'P': options.platform = optarg; break;
		case 'i': options.run_id = optarg; break;
		case 'c': options.condition = optarg; break;
		case 'o': options.output = optarg; break;
		case 'D': options.device = optarg; break;
		default: usage(argv[0]); return 2;
		}
	}
	if (!options.output || !options.iterations ||
	    !safe_label(options.platform) || !safe_label(options.run_id) ||
	    !safe_label(options.condition)) {
		usage(argv[0]);
		return 2;
	}

	fd = open(options.device, O_RDWR | O_CLOEXEC);
	if (fd < 0) { perror(options.device); return 1; }
	if (ioctl(fd, XHYPASS_IOC_GET_INFO, &info)) {
		perror("GET_INFO"); rc = -errno; goto out;
	}
	if (info.abi_version != XHYPASS_IOCTL_ABI_VERSION) {
		fprintf(stderr, "ABI mismatch: module=%u userspace=%u\n",
			info.abi_version, XHYPASS_IOCTL_ABI_VERSION);
		rc = -EPROTO; goto out;
	}
	if (info.mode != XHYPASS_GUEST_DYN) {
		fprintf(stderr, "self-test must start in DYN mode\n");
		rc = -EINVAL; goto out;
	}
	if (ioctl(fd, XHYPASS_IOC_RESET_STATS)) {
		perror("RESET_STATS"); rc = -errno; goto out;
	}
	memset(&config, 0, sizeof(config));
	config.producer_cpu = options.producer_cpu;
	config.timer_delay_us = options.timer_delay_us;
	config.event_timeout_ms = options.event_timeout_ms;
	config.source_mask = options.source_mask;
	config.spi_delay_us = options.spi_delay_us;
	if (ioctl(fd, XHYPASS_IOC_START_SELFTEST, &config)) {
		perror("START_SELFTEST"); rc = -errno; goto out;
	}
	selftest_started = true;

	for (completed = 0; completed < options.iterations; completed++) {
		result = request_mode(fd, XHYPASS_IOC_ENTER_RTO,
			options.max_retries, options.retry_delay_us, &enter_retries);
		if (result) { rc = result; break; }
		if (options.dwell_us) usleep(options.dwell_us);
		result = request_mode(fd, XHYPASS_IOC_EXIT_RTO,
			options.max_retries, options.retry_delay_us, &exit_retries);
		if (result) { rc = result; break; }
		if (options.dwell_us) usleep(options.dwell_us);
		if (!((completed + 1) % 100) || completed + 1 == options.iterations) {
			fprintf(stderr, "XHYPASS_PROGRESS pairs=%lu/%lu source_mask=%u\n",
				completed + 1, options.iterations, options.source_mask);
			fflush(stderr);
		}
	}

	if (!ioctl(fd, XHYPASS_IOC_GET_INFO, &info) &&
	    info.mode == XHYPASS_GUEST_RTO)
		(void)request_mode(fd, XHYPASS_IOC_EXIT_RTO,
			options.max_retries, options.retry_delay_us, &exit_retries);
	if (ioctl(fd, XHYPASS_IOC_STOP_SELFTEST) && !rc) rc = -errno;
	selftest_started = false;
	if (ioctl(fd, XHYPASS_IOC_GET_SELFTEST_STATS, &selftest_stats) && !rc)
		rc = -errno;
	if (ioctl(fd, XHYPASS_IOC_GET_STATS, &guest_stats) && !rc) rc = -errno;
	if (ioctl(fd, XHYPASS_IOC_GET_INFO, &info) && !rc) rc = -errno;
	result = write_result(&options, &info, &guest_stats, &selftest_stats,
		completed, enter_retries, exit_retries, rc);
	if (!rc) rc = result;

out:
	if (selftest_started) (void)ioctl(fd, XHYPASS_IOC_STOP_SELFTEST);
	if (fd >= 0) close(fd);
	if (rc) {
		errno = -rc;
		perror("xhypass correctness run");
		return 1;
	}
	return 0;
}
