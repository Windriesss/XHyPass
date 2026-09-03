"""Shared publication style for every XHyPass Matplotlib figure."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl


PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "axes.titleweight": "normal",
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    # Embed TrueType outlines. Matplotlib's default value 3 emits Type 3 fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.use14corefonts": False,
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_paper_style(overrides: Mapping[str, object] | None = None) -> None:
    """Apply the shared style, optionally followed by layout-only overrides."""
    settings = dict(PAPER_RC)
    if overrides:
        settings.update(overrides)
    mpl.rcParams.update(settings)
