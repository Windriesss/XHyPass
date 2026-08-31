# Plotting scripts

Run `python reproduce_figures.py` from the repository root to recreate all six
released PDFs. Individual scripts may also be run directly with Python. They
resolve input and output locations from their own file paths and do not depend
on the caller's previous workstation directory layout.

The common environment marker scheme is defined in
`plot_nn_scatter_2x3.py`. Related Xen configurations use one marker shape, with
filled and hollow markers distinguishing the native-WFx setting. Motivation and
neural-network plots reuse the same marker helpers so legends and data points
remain consistent.
