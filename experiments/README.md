# Experiment reproduction

The experiment controller is in `automation/`, guest-side scripts are in
`workloads/`, and the HTTP request generator is in `loadgen/`.

## Host requirements

- Python 3.10 or newer with `pyserial` and `paramiko`
- Serial access to the target board
- SSH/SFTP access where required by the Jailhouse and Xen setups
- Board-specific boot images and device trees

Install the Python packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run automation entry points from the repository root. New results are written
below `data/RK3588/` or `data/E2000Q/`. Before using real hardware, review the
serial ports, boot commands, IP addresses, CPU assignments, and timeouts in
`automation/config/` and in the selected entry point.

Site-specific boot files belong in `experiments/boot-artifacts/`. Files staged
for a board or TFTP/SFTP service belong in `experiments/deploy/`; both locations
are ignored by Git. Do not commit credentials or proprietary firmware images.

## Load generator

Build on a Linux host with libcurl development headers:

```bash
make -C experiments/loadgen
```

Example:

```bash
experiments/loadgen/loadgen \
  --url http://TARGET:PORT/infer \
  --duration 600 \
  --constant-qps 1 \
  --qps 2 \
  --peak-qps 8 \
  --period-sec 60 \
  --workers 8 \
  --seed 12345 \
  --out results.csv
```

The generator combines constant, Poisson, and sinusoidal burst arrivals and
writes request timestamps, end-to-end response times, reported inference times,
and request identifiers as CSV.
