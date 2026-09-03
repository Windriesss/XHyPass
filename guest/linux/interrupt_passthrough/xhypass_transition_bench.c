// SPDX-License-Identifier: GPL-2.0

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "xhypass_ioctl.h"

#define CSV_SCHEMA_VERSION 1

struct options {
	const char *device;
	const char *output;
	const char *run_id;
	const char *condition;
	uint64_t iterations;
	unsigned int max_retries;
	unsigned int retry_delay_us;
	unsigned int dwell_us;
};

static uint64_t counter_frequency(void)
{
#ifdef __aarch64__
	uint64_t value;

	asm volatile("mrs %0, cntfrq_el0" : "=r" (value));
	return value;
#else
	return UINT64_C(1000000000);
#endif
}

static uint64_t counter_value(void)
{
#ifdef __aarch64__
	uint64_t value;

	asm volatile("isb; mrs %0, cntvct_el0; isb" : "=r" (value) :: "memory");
	return value;
#else
	struct timespec now;

	clock_gettime(CLOCK_MONOTONIC_RAW, &now);
	return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
#endif
}

static uint64_t cycles_to_ns(uint64_t cycles, uint64_t frequency)
{
	return (uint64_t)(((__uint128_t)cycles * UINT64_C(1000000000)) /
			  frequency);
}

static void usage(FILE *stream, const char *program)
{
	fprintf(stream,
		"Usage: %s [options]\n"
		"  --iterations N       successful DYN/RTO pairs (default 10000)\n"
		"  --output PATH        raw attempt CSV, or - for stdout\n"
		"  --run-id ID          independent-run identifier\n"
		"  --condition NAME     idle, memory-pressure, spi-storm, ...\n"
		"  --max-retries N      retries per direction (default 1000)\n"
		"  --retry-delay-us N   delay after EBUSY/EAGAIN (default 0)\n"
		"  --dwell-us N         time spent in each stable mode\n"
		"  --device PATH        ioctl device (default /dev/xhypass)\n",
		program);
}

static uint64_t parse_u64(const char *value, const char *name)
{
	char *end = NULL;
	unsigned long long parsed;

	errno = 0;
	parsed = strtoull(value, &end, 10);
	if (errno || !end || *end)
		fprintf(stderr, "invalid %s: %s\n", name, value), exit(EXIT_FAILURE);
	return (uint64_t)parsed;
}

static struct options parse_options(int argc, char **argv)
{
	struct options options = {
		.device = XHYPASS_DEVICE_PATH,
		.output = "transition_attempts.csv",
		.run_id = "run-001",
		.condition = "idle",
		.iterations = 10000,
		.max_retries = 1000,
	};
	static const struct option long_options[] = {
		{ "iterations", required_argument, NULL, 'n' },
		{ "output", required_argument, NULL, 'o' },
		{ "run-id", required_argument, NULL, 'r' },
		{ "condition", required_argument, NULL, 'c' },
		{ "max-retries", required_argument, NULL, 'm' },
		{ "retry-delay-us", required_argument, NULL, 'd' },
		{ "dwell-us", required_argument, NULL, 'w' },
		{ "device", required_argument, NULL, 'D' },
		{ "help", no_argument, NULL, 'h' },
		{ NULL, 0, NULL, 0 },
	};
	int option;

	while ((option = getopt_long(argc, argv, "n:o:r:c:m:d:w:D:h",
				     long_options, NULL)) != -1) {
		switch (option) {
		case 'n': options.iterations = parse_u64(optarg, "iterations"); break;
		case 'o': options.output = optarg; break;
		case 'r': options.run_id = optarg; break;
		case 'c': options.condition = optarg; break;
		case 'm': options.max_retries = parse_u64(optarg, "max-retries"); break;
		case 'd': options.retry_delay_us = parse_u64(optarg, "retry-delay-us"); break;
		case 'w': options.dwell_us = parse_u64(optarg, "dwell-us"); break;
		case 'D': options.device = optarg; break;
		case 'h': usage(stdout, argv[0]); exit(EXIT_SUCCESS);
		default: usage(stderr, argv[0]); exit(EXIT_FAILURE);
		}
	}
	if (!options.iterations)
		fprintf(stderr, "iterations must be positive\n"), exit(EXIT_FAILURE);
	return options;
}

static int request_mode(int fd, unsigned long command, const char *direction,
			FILE *output, const struct options *options,
			uint64_t iteration, uint64_t frequency)
{
	unsigned int attempt;
	uint64_t request_start = counter_value();

	for (attempt = 0; attempt <= options->max_retries; attempt++) {
		uint64_t before = counter_value();
		int rc = ioctl(fd, command);
		int saved_errno = rc ? errno : 0;
		uint64_t after = counter_value();
		uint64_t cycles = after - before;
		uint64_t request_cycles = after - request_start;
		int result = rc ? -saved_errno : 0;

		fprintf(output, "%d,%s,%s,%" PRIu64 ",%s,%u,%d,%" PRIu64
			",%" PRIu64 ",%" PRIu64 "\n",
			CSV_SCHEMA_VERSION, options->run_id,
			options->condition, iteration, direction, attempt, result,
			cycles, cycles_to_ns(cycles, frequency),
			cycles_to_ns(request_cycles, frequency));

		if (!result)
			return 0;
		if (result != -EBUSY && result != -EAGAIN)
			return result;
		if (attempt == options->max_retries)
			return result;
		if (options->retry_delay_us)
			usleep(options->retry_delay_us);
	}
	return -EIO;
}

int main(int argc, char **argv)
{
	struct options options = parse_options(argc, argv);
	struct xhypass_guest_info info;
	uint64_t frequency = counter_frequency();
	FILE *output;
	uint64_t iteration;
	bool in_rto = false;
	int fd;
	int rc = EXIT_SUCCESS;

	output = !strcmp(options.output, "-") ? stdout : fopen(options.output, "w");
	if (!output) {
		perror(options.output);
		return EXIT_FAILURE;
	}

	fd = open(options.device, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		perror(options.device);
		rc = EXIT_FAILURE;
		goto out_close_output;
	}
	if (ioctl(fd, XHYPASS_IOC_GET_INFO, &info)) {
		perror("XHYPASS_IOC_GET_INFO");
		rc = EXIT_FAILURE;
		goto out_close_device;
	}
	if (info.abi_version != XHYPASS_IOCTL_ABI_VERSION) {
		fprintf(stderr, "unsupported ioctl ABI: %u\n", info.abi_version);
		rc = EXIT_FAILURE;
		goto out_close_device;
	}
	if (info.mode != XHYPASS_GUEST_DYN) {
		fprintf(stderr, "benchmark must start in DYN mode\n");
		rc = EXIT_FAILURE;
		goto out_close_device;
	}

	fprintf(output, "schema_version,run_id,condition,iteration,direction,"
		"attempt,rc,counter_cycles,duration_ns,request_elapsed_ns\n");
	for (iteration = 0; iteration < options.iterations; iteration++) {
		int result = request_mode(fd, XHYPASS_IOC_ENTER_RTO, "DYN-to-RTO",
					  output, &options, iteration, frequency);
		if (result) {
			fprintf(stderr, "entry failed at iteration %" PRIu64 ": %d\n",
				iteration, result);
			rc = EXIT_FAILURE;
			break;
		}
		in_rto = true;
		if (options.dwell_us)
			usleep(options.dwell_us);

		result = request_mode(fd, XHYPASS_IOC_EXIT_RTO, "RTO-to-DYN",
				      output, &options, iteration, frequency);
		if (result) {
			fprintf(stderr, "exit failed at iteration %" PRIu64 ": %d\n",
				iteration, result);
			rc = EXIT_FAILURE;
			break;
		}
		in_rto = false;
		if (options.dwell_us)
			usleep(options.dwell_us);
		if (!((iteration + 1) % 100) || iteration + 1 == options.iterations) {
			fprintf(stderr, "XHYPASS_LATENCY_PROGRESS pairs=%" PRIu64
				"/%" PRIu64 "\n", iteration + 1, options.iterations);
			fflush(stderr);
			fflush(output);
		}
	}

	if (in_rto) {
		int cleanup = ioctl(fd, XHYPASS_IOC_EXIT_RTO);

		if (cleanup)
			fprintf(stderr, "warning: cleanup exit failed: %s\n",
				strerror(errno));
	}

out_close_device:
	close(fd);
out_close_output:
	if (output != stdout && fclose(output)) {
		perror("fclose");
		rc = EXIT_FAILURE;
	}
	return rc;
}
