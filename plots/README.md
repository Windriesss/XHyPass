# Plotting scripts

Run `python reproduce_figures.py` from the repository root to recreate the six
released PDFs. The public plotting entry points are intentionally limited to:

| Output | Entry point |
| --- | --- |
| `motivation_tradeoff.pdf` | `plot_motivation_tradeoff.py` |
| `interrupt_latency_runmax.pdf` | `plot_interrupt_latency_runmax.py` |
| `interrupt_latency_tail_percentiles.pdf` | `plot_interrupt_latency_tail_percentiles.py` |
| `cyclictest_runmax.pdf` | `plot_cyclictest_runmax.py` |
| `rk3588_nn_metrics_scatter_2x3.pdf` | `plot_nn_scatter_2x3.py` |
| `rk3588_nn_comprehensive_4x3.pdf` | `analyze_nn.py` |

Individual entry points may also be run directly with Python. They resolve
input and output locations from their own file paths and do not depend on the
caller's previous workstation directory layout. Files under `xhypass_plot/`
and `nn_mixed_workload_stats_core.py` are shared parsing, statistics, and style
modules rather than additional figure entry points.

All Matplotlib entry points use `xhypass_plot/style.py`. The shared style
embeds TrueType outlines in PDF/PS output and disables Type 3 font generation.
`reproduce_figures.py` runs `tools/check_pdf_fonts.py` after rendering and
fails if any released PDF contains a Type 3 font.

The common environment marker scheme is defined in
`plot_nn_scatter_2x3.py`. Related Xen configurations use one marker shape, with
filled and hollow markers distinguishing the native-WFx setting. Motivation and
neural-network plots reuse the same marker helpers so legends and data points
remain consistent.
