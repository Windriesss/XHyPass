# Platform configuration

- `TL3588/lab.json` is the active TL3588 configuration used by the existing
  experiment scripts.
- `E2000Q/lab.json` contains the same seven environment slots but is disabled
  until its serial port, U-Boot commands, network addresses, boot artifacts,
  CPU topology, Jailhouse cells, and Xen commands are filled in.

Experiment data follows the matching `data/<platform>/...` hierarchy.
