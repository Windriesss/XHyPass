#!/usr/bin/env python3
"""Edit the settings below, then run this file without command-line arguments."""

from pathlib import Path

try:
    from smoke_scripts._project import PROJECT_ROOT
except ModuleNotFoundError:
    from _project import PROJECT_ROOT

from xhypass_lab.config import resolved_run_config
from xhypass_lab.platforms import load_platform_config
from xhypass_lab.runner import ExperimentRunner


# ============================================================================
# User settings: normally, only edit this section.
# ============================================================================

# Target environment and experiment.
ENVIRONMENT = "jailhouse"     # bare / jailhouse / xen_credit2 / XHyPass / xen_credit2_WFX / xen_null / xen_null_WFX
EXPERIMENT = "cyclictest-stress"  # cyclictest / cyclictest-stress

# Serial console.
SERIAL_PORT = "COM10"
SERIAL_BAUDRATE = 115_200

# U-Boot command for the selected environment.
# Fill in the corresponding command when testing other systems.
UBOOT_COMMANDS = {
    "bare": "run boot_oee",
    "jailhouse": "run boot_oee",
    "xen_credit2": "run boot_xen",
    "xen_credit2_WFX": "run boot_xen",
    "XHyPass": "run boot_xen",
    "xen_null": "run boot_xen",
    "xen_null_WFX": "run boot_xen",
}

BOOT_MAX_ATTEMPTS = 3
BOOT_RETRY_DELAY_SECONDS = 1.0

# Global reboot recovery. The previous run may have left Jailhouse active even
# when the next target environment is bare/Xen/XHyPass, so every reboot probes
# this root-cell SSH endpoint first.
REBOOT_SSH_HOST = "202.197.67.50"
REBOOT_SSH_PORT = 22
REBOOT_SSH_USERNAME = "root"
REBOOT_SSH_PASSWORD = ""
REBOOT_SSH_KEY_FILE = None
REBOOT_SSH_TIMEOUT = 8
ALLOW_SERIAL_REBOOT_FALLBACK = False

# Jailhouse root-cell network and SSH settings.
JAILHOUSE_ROOTCELL_IP = "202.197.67.50"
JAILHOUSE_NETMASK = None       # e.g. "255.255.255.0"; None uses ifconfig IP only
JAILHOUSE_MODULE_LOAD_COMMAND = "insmod jailhouse.ko"
JAILHOUSE_MODULE_SETTLE_SECONDS = 20
ROOTCELL_SSH_USERNAME = "root"
ROOTCELL_SSH_PASSWORD = ""     # set it here if root-cell SSH requires a password
ROOTCELL_SSH_KEY_FILE = None    # e.g. Path("credentials") / "id_ed25519"
ROOTCELL_SSH_CONNECT_TIMEOUT = 60

# xen_credit2 host boot files, dom0 setup, and SSH settings.
# Resolve these paths relative to this script so moving the project does not
# require editing absolute drive paths again.
RK3588_BOOT_ENV_ROOT = PROJECT_ROOT.parent / "boot-artifacts" / "RK3588"
XEN_CREDIT2_BOOT_SOURCE = RK3588_BOOT_ENV_ROOT / "Xen_credit2"
XEN_CREDIT2_NATIVEWFX_BOOT_SOURCE = (
    RK3588_BOOT_ENV_ROOT / "Xen_credit2_nativeWFX"
)
XHYPASS_BOOT_SOURCE = RK3588_BOOT_ENV_ROOT / "XHyPass"
XEN_NULL_BOOT_SOURCE = RK3588_BOOT_ENV_ROOT / "Xen_null"
XEN_NULL_WFX_BOOT_SOURCE = RK3588_BOOT_ENV_ROOT / "Xen_null_nativeWFX"
XEN_BOOT_SOURCES = {
    "xen_credit2": XEN_CREDIT2_BOOT_SOURCE,
    "xen_credit2_WFX": XEN_CREDIT2_NATIVEWFX_BOOT_SOURCE,
    "XHyPass": XHYPASS_BOOT_SOURCE,
    "xen_null": XEN_NULL_BOOT_SOURCE,
    "xen_null_WFX": XEN_NULL_WFX_BOOT_SOURCE,
}
XEN_BOOT_TARGET = PROJECT_ROOT.parent / "deploy" / "RK3588"
XEN_DOM0_IP = "202.197.67.50"
XEN_DOM0_SSH_USERNAME = "root"
XEN_DOM0_SSH_PASSWORD = ""
XEN_DOM0_SSH_KEY_FILE = None
XEN_DOM0_SSH_CONNECT_TIMEOUT = 60
XEN_DOM0_INIT_COMMANDS = [
    "cd ~/dom-interrupt",
    "sh set_bridge.sh",
    "xl create compute_NN_dom1.cfg",
]
XEN_DOM0_PIN_COMMAND = (
    "xl vcpu-pin 0 0 0-5; xl vcpu-pin 0 1 0-5; xl vcpu-pin 0 2 0-5; "
    "xl vcpu-pin 0 3 0-5; xl vcpu-pin 0 4 0-5; xl vcpu-pin 0 5 0-5; "
    "xl vcpu-pin 0 6 6"
)
XEN_DOM1_PIN_COMMAND = (
    "xl vcpu-pin dom1 0 0-5; xl vcpu-pin dom1 1 0-5; "
    "xl vcpu-pin dom1 2 0-5; xl vcpu-pin dom1 3 0-5; "
    "xl vcpu-pin dom1 4 0-5; xl vcpu-pin dom1 5 0-5; "
    "xl vcpu-pin dom1 6 7"
)
XEN_PIN_VERIFY_COMMANDS = ["xl vcpu-list 0", "xl vcpu-list dom1"]
XEN_DOM1_CONSOLE_COMMAND = "xl console dom1"
XEN_DOM1_USERNAME = "root"
XEN_DOM1_PASSWORD = ""
XEN_DOM1_BOOT_TIMEOUT = 180
XEN_POST_DOM1_LOGIN_DELAY_SECONDS = 10

JAILHOUSE_ENABLE_COMMAND = (
    "jailhouse enable /usr/share/jailhouse/cells/tl3588.cell"
)
JAILHOUSE_LINUX_COMMAND = (
    "jailhouse cell linux /usr/share/jailhouse/cells/tl3588-linux-demo.cell "
    "./Image -d /usr/share/jailhouse/cells/dts/inmate-tl3588.dtb "
    "-i new_phytium_rootfs.cpio.gz -c \"isolcpus=3 rcu_nocb=3\""
)
NONROOT_USERNAME = "root"
NONROOT_PASSWORD = ""
NONROOT_BOOT_TIMEOUT = 180

# Repetition and reboot behavior.
RUNS = 1
REBOOT_POLICY = "each-run"    # each-run: reboot before every run; once: boot once
PLATFORM = "RK3588"
DATA_ROOT = PROJECT_ROOT.parents[1] / "data" / PLATFORM

# False means actually open COM10 and run the experiment. Change to True only
# when you want to inspect the plan without touching the board.
DRY_RUN = False

# Print the board's serial output in this terminal while also saving a full log.
SHOW_SERIAL_OUTPUT = True

# cyclictest parameters. For a quick check, use 10 seconds. For final data,
# change DURATION_SECONDS to 600 and increase RUNS as needed.
CYCLICTEST_BINARY = "./cyclictest"
CPU = 6
THREADS = 1
PRIORITY = 99
INTERVAL_US = 1_000
DURATION_SECONDS = 10
HISTOGRAM_LIMIT_US = 10_000
GRACE_SECONDS = 60
CLEANUP_REMOTE_FILES = True

# stress-ng parameters used only by the cyclictest-stress experiment. Its
# timeout always equals DURATION_SECONDS.
STRESS_BINARY = "./stress-ng"
STRESS_CPUS = "6"
STRESS_VM_WORKERS = 1
STRESS_VM_BYTES = "256M"

# Boot/login timeouts in seconds. Increase BOOT_TIMEOUT for a slower image.
PROBE_TIMEOUT = 2
UBOOT_TIMEOUT = 60
BOOT_TIMEOUT = 180
LOGIN_TIMEOUT = 30

# ============================================================================
# Framework glue: no changes should normally be needed below this line.
# ============================================================================


def build_run_config() -> dict:
    config = load_platform_config(PLATFORM)

    config["serial"]["port"] = SERIAL_PORT
    config["serial"]["baudrate"] = SERIAL_BAUDRATE
    config["serial"]["show_output"] = SHOW_SERIAL_OUTPUT
    config["reboot"] = {
        "ssh": {
            "host": REBOOT_SSH_HOST,
            "port": REBOOT_SSH_PORT,
            "username": REBOOT_SSH_USERNAME,
            "password": REBOOT_SSH_PASSWORD,
            "key_file": REBOOT_SSH_KEY_FILE,
            "connect_timeout": REBOOT_SSH_TIMEOUT,
            "retry_interval": 1,
            "allow_agent": True,
            "look_for_keys": True,
            "progress_interval": 10,
        },
        "allow_serial_fallback": ALLOW_SERIAL_REBOOT_FALLBACK,
        "nonroot_login_prompt": r"[A-Za-z0-9_.-]+ login:\s*$",
        "nonroot_username": NONROOT_USERNAME,
        "nonroot_password": NONROOT_PASSWORD,
        "nonroot_cmdline_markers": ["isolcpus=3", "rcu_nocb=3"],
    }
    config["environments"][ENVIRONMENT]["boot_command"] = UBOOT_COMMANDS[ENVIRONMENT]
    config["environments"][ENVIRONMENT]["boot_max_attempts"] = BOOT_MAX_ATTEMPTS
    config["environments"][ENVIRONMENT]["boot_retry_delay_seconds"] = BOOT_RETRY_DELAY_SECONDS
    if ENVIRONMENT == "jailhouse":
        config["environments"][ENVIRONMENT].update(
            {
                 "rootcell_ip": JAILHOUSE_ROOTCELL_IP,
                 "rootcell_netmask": JAILHOUSE_NETMASK,
                 "module_load_command": JAILHOUSE_MODULE_LOAD_COMMAND,
                 "module_settle_seconds": JAILHOUSE_MODULE_SETTLE_SECONDS,
                 "ssh": {
                    "host": JAILHOUSE_ROOTCELL_IP,
                    "port": 22,
                    "username": ROOTCELL_SSH_USERNAME,
                    "password": ROOTCELL_SSH_PASSWORD,
                    "key_file": ROOTCELL_SSH_KEY_FILE,
                    "connect_timeout": ROOTCELL_SSH_CONNECT_TIMEOUT,
                    "retry_interval": 2,
                    "allow_agent": True,
                    "look_for_keys": True,
                    "progress_interval": 10,
                },
                "enable_command": JAILHOUSE_ENABLE_COMMAND,
                "linux_command": JAILHOUSE_LINUX_COMMAND,
                "nonroot_username": NONROOT_USERNAME,
                "nonroot_password": NONROOT_PASSWORD,
                "nonroot_login_prompt": r"[A-Za-z0-9_.-]+ login:\s*$",
                "nonroot_boot_timeout": NONROOT_BOOT_TIMEOUT,
                "nonroot_cmdline_markers": ["isolcpus=3", "rcu_nocb=3"],
            }
        )
    elif ENVIRONMENT in XEN_BOOT_SOURCES:
        # Apply editable common Xen settings to the parent. Variant-specific
        # commands in child environments continue to override these defaults.
        config["environments"]["xen_credit2"].update(
            {
                "dom0_init_commands": list(XEN_DOM0_INIT_COMMANDS),
                "ssh": {
                    "host": XEN_DOM0_IP,
                    "port": 22,
                    "username": XEN_DOM0_SSH_USERNAME,
                    "password": XEN_DOM0_SSH_PASSWORD,
                    "key_file": XEN_DOM0_SSH_KEY_FILE,
                    "connect_timeout": XEN_DOM0_SSH_CONNECT_TIMEOUT,
                    "retry_interval": 2,
                    "allow_agent": True,
                    "look_for_keys": True,
                    "progress_interval": 10,
                },
                "dom0_pin_command": XEN_DOM0_PIN_COMMAND,
                "dom1_pin_command": XEN_DOM1_PIN_COMMAND,
                "pin_verify_commands": list(XEN_PIN_VERIFY_COMMANDS),
                "dom1_console_command": XEN_DOM1_CONSOLE_COMMAND,
                "dom1_login_prompt": r"[A-Za-z0-9_.-]+ login:\s*$",
                "dom1_username": XEN_DOM1_USERNAME,
                "dom1_password": XEN_DOM1_PASSWORD,
                "dom1_shell_prompts": [r"root@[^\r\n]*#\s*$", r"#\s*$"],
                "dom1_boot_timeout": XEN_DOM1_BOOT_TIMEOUT,
                "post_dom1_login_delay_seconds": XEN_POST_DOM1_LOGIN_DELAY_SECONDS,
            }
        )
        config["environments"][ENVIRONMENT].update(
            {
                "local_boot_files": {
                    "source_dir": str(XEN_BOOT_SOURCES[ENVIRONMENT]),
                    "target_dir": str(XEN_BOOT_TARGET),
                },
            }
        )
    config["timeouts"].update(
        {
            "probe": PROBE_TIMEOUT,
            "uboot": UBOOT_TIMEOUT,
            "boot": BOOT_TIMEOUT,
            "login": LOGIN_TIMEOUT,
        }
    )

    run_config = resolved_run_config(config, ENVIRONMENT, EXPERIMENT, {})
    run_config["experiment"].update(
        {
            "binary": CYCLICTEST_BINARY,
            "cpu": CPU,
            "threads": THREADS,
            "priority": PRIORITY,
            "interval_us": INTERVAL_US,
            "duration_seconds": DURATION_SECONDS,
            "histogram_limit_us": HISTOGRAM_LIMIT_US,
            "grace_seconds": GRACE_SECONDS,
            "cleanup_remote": CLEANUP_REMOTE_FILES,
            "stress_binary": STRESS_BINARY,
            "stress_cpus": STRESS_CPUS,
            "stress_vm_workers": STRESS_VM_WORKERS,
            "stress_vm_bytes": STRESS_VM_BYTES,
        }
    )
    run_config["experiment"].update(
        run_config["environment"].get("experiment_overrides", {})
    )
    return run_config


def main() -> int:
    if RUNS < 1:
        raise ValueError("RUNS must be at least 1")
    if REBOOT_POLICY not in {"once", "each-run"}:
        raise ValueError("REBOOT_POLICY must be 'once' or 'each-run'")

    runner = ExperimentRunner(build_run_config(), DATA_ROOT, dry_run=DRY_RUN)
    outputs = runner.run(RUNS, REBOOT_POLICY)
    if not DRY_RUN:
        print("Completed:")
        for path in outputs:
            print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
