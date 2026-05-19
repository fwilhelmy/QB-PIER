"""Clean public entry points for gradient-related training analytics.

This module groups the previous gradient helpers without rewriting their
implementation. Keep the lower-level modules until downstream notebooks/imports
are updated.
"""

from .gradient_flow import generate_gradient_flow_plots, plot_gradient_flow, try_plot_gradient_flow
from .gradient_map import generate_gradient_map_plots, plot_gradient_map, try_plot_gradient_map
from .gradient_norms import gradient_norms

__all__ = [
    "generate_gradient_flow_plots",
    "plot_gradient_flow",
    "try_plot_gradient_flow",
    "generate_gradient_map_plots",
    "plot_gradient_map",
    "try_plot_gradient_map",
    "gradient_norms",
]
