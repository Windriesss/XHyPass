# SPDX-License-Identifier: GPL-2.0

obj-m += interrupt_passthrough.o

KERNEL_SRC ?= /lib/modules/$(shell uname -r)/build
MODULE_DIR := $(CURDIR)
USER_CC := $(CROSS_COMPILE)gcc
USER_CFLAGS ?= -O2 -Wall -Wextra -Werror -std=gnu11

.PHONY: all module user clean

all: module user

module:
	$(MAKE) -C $(KERNEL_SRC) M=$(MODULE_DIR) modules

user: xhypass_transition_bench xhypass_correctness

xhypass_transition_bench: xhypass_transition_bench.c xhypass_ioctl.h
	$(USER_CC) $(USER_CFLAGS) -o $@ xhypass_transition_bench.c

xhypass_correctness: xhypass_correctness.c xhypass_ioctl.h
	$(USER_CC) $(USER_CFLAGS) -o $@ xhypass_correctness.c

clean:
	$(MAKE) -C $(KERNEL_SRC) M=$(MODULE_DIR) clean
	$(RM) xhypass_transition_bench xhypass_correctness
