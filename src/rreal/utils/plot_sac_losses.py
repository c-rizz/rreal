#!/usr/bin/env python3

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
import argparse

# Needs: uv pip install "plotly-resampler==0.10.0" "dash>=2.14,<3" "plotly>=5.18,<6"
#

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Path to binary losses file")
parser.add_argument("--remote", action="store_true", help="Allow remote access to the dashboard")
parser.add_argument("--logy", action="store_true", help="Use logarithmic scale for all y-axes")
parser.add_argument("--logq", action="store_true", help="Use log scale for q_loss subplot")
parser.add_argument("--logactor", action="store_true", help="Use log scale for actor_loss subplot (shifts to positive)")
parser.add_argument("--logalpha", action="store_true", help="Use log scale foralpha subplot")
args = parser.parse_args()

dt = np.dtype([("q_loss", "f4"), ("actor_loss", "f4"), ("alpha_loss", "f4"), ("alpha", "f4")])
arr = np.fromfile(args.i, dtype=dt)
steps = np.arange(len(arr))

names: tuple[str, ...] = arr.dtype.names  # type: ignore[assignment]

base_fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True,
                         vertical_spacing=0.05,
                         subplot_titles=names)
fig = FigureResampler(base_fig, default_n_shown_samples=2000)

for i, name in enumerate(names, start=1):
    y = np.ascontiguousarray(arr[name])
    display_name = name
    if (args.logactor and name == "actor_loss"):
        offset = float(np.abs(y.min())) + 1e-6
        y = y + offset
        display_name = f"{name} (shifted +{offset:.3g})"
    fig.add_trace(
        go.Scattergl(name=display_name, mode="lines"),
        hf_x=steps,
        hf_y=y,
        row=i, col=1,
    )

layout = {"hovermode": "x unified"}
for i, name in enumerate(names, start=1):
    use_log = args.logy or (args.logq and name == "q_loss") or (args.logactor and name == "actor_loss") or (args.logalpha and name =="alpha")
    if use_log:
        yaxis_key = "yaxis" if i == 1 else f"yaxis{i}"
        layout[yaxis_key] = {"type": "log"}

fig.update_layout(**layout)
fig.show_dash(mode="external", port=8050, host="0.0.0.0" if args.remote else "127.0.0.1",
              graph_properties={"style": {"height": "100vh"}})
input("Press enter to exit...")
