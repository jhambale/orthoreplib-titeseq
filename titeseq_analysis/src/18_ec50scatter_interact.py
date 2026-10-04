#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2026 May 18
@author: jhambale

18_ec50scatter_interact.py
==================
Intakes input library of sequences and associated EC50 values, displays the EC50
values for each reference as a scatter plot, which is interactive and allows the
user to hover over points to see the EC50 values of sequences close to a clicked point.

-----
Usage
-----
python 18_ec50scatter_interact.py \\
    -i ../../titeseq_data/outputs/jh_002/tite_seq_Kds2/cluster_eval_tite_seq_Kds_passed_variants/tite_seq_clustered_tite_seq_Kds_passed_variants.csv \\
"""

import argparse
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, no_update

# Global variables to store the dataframe and base colors
df = None
base_colors = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Intakes input library of sequences and associated EC50 values, "
            "displays the EC50 values for each reference as a scatter plot, "
            "which is interactive and allows the user to hover over points to "
            "see the EC50 values of sequences close to a clicked point."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i",
        required=True,
        metavar="CSV",
        help=(
            "Path to input CSV containing at minimum 'aa_sequence', 'variant_id', "
            "'log10_rep1_Kd', 'log10_rep2_Kd', and optionally 'predicted_category' columns."
        ),
    )
    return p


def hamming(a, b):
    if len(a) != len(b):
        return np.inf
    return sum(c1 != c2 for c1, c2 in zip(a, b))


# Color map for binding categories
color_map = {
    "Tight Binder": "#2e8b57",
    "Moderate Binder": "#4a78b5",
    "Weak Binder": "#888888",
}


def build_figure(selected_idx=None):
    sizes = np.full(len(df), 9)
    opac = np.full(len(df), 0.85)
    colors = (
        df.get("predicted_category", pd.Series(["Moderate Binder"] * len(df)))
        .map(color_map)
        .fillna("#4a78b5")
    ).to_numpy()

    if selected_idx is not None:
        sel_seq = df.loc[selected_idx, "aa_sequence"]
        dists = df["aa_sequence"].apply(lambda s: hamming(sel_seq, s)).to_numpy()
        dim_mask = ~np.isin(dists, [0, 1, 2])
        opac[dim_mask] = 0.08
        sizes[selected_idx] = 18

    fig = go.Figure(
        go.Scattergl(
            x=df["log10_rep1_Kd"],
            y=df["log10_rep2_Kd"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=opac,
                line=dict(width=0.5, color="black"),
            ),
            customdata=np.stack([df.index, df["variant_id"]], axis=-1),
            hovertemplate="variant=%{customdata[1]}<br>rep1=%{x:.2f}<br>rep2=%{y:.2f}<extra></extra>",
        )
    )
    lo, hi = 0.8, 3.4
    fig.add_shape(
        type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="blue")
    )
    fig.update_layout(
        xaxis_title="log10(Kd) rep1 [nM]",
        yaxis_title="log10(Kd) rep2 [nM]",
        width=800,
        height=750,
        template="simple_white",
        hoverlabel=dict(namelength=-1, showarrow=False),
    )
    return fig


app = Dash(__name__, suppress_callback_exceptions=True)

# Placeholder layout - populated after data is loaded
app.layout = html.Div(id="app-container")


@app.callback(
    Output("app-container", "children"),
    Input("app-container", "id"),
)
def load_layout(_):
    """Build the full layout once data is loaded."""
    if df is None:
        return html.Div("Loading data...")
    return html.Div(
        [
            dcc.Graph(id={"type": "scatter", "index": 0}, figure=build_figure()),
            html.Div(
                id={"type": "info", "index": 0},
                style={"fontFamily": "monospace", "padding": "8px"},
            ),
            html.Button(
                "Export Highlighted Sequences",
                id="export-button",
                n_clicks=0,
                style={
                    "padding": "10px 20px",
                    "margin": "10px",
                    "fontSize": "16px",
                    "backgroundColor": "#4a78b5",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "4px",
                    "cursor": "pointer",
                },
            ),
            dcc.Download(id="download-csv"),
            html.Div(id="export-status", style={"padding": "8px", "color": "green"}),
            dcc.Store(id="selected-idx-store", data=None),
        ]
    )


@app.callback(
    Output({"type": "scatter", "index": ALL}, "figure"),
    Output({"type": "info", "index": ALL}, "children"),
    Output("selected-idx-store", "data"),
    Input({"type": "scatter", "index": ALL}, "clickData"),
    prevent_initial_call=True,
)
def on_click(clickData):
    if not clickData or not clickData[0]:
        return (
            [no_update],
            ["Click any dot to highlight it and dim variants >2 AA away."],
            None,
        )
    idx = int(clickData[0]["points"][0]["customdata"][0])
    seq = df.loc[idx, "aa_sequence"]
    return (
        [build_figure(idx)],
        [f"Selected idx={idx}  seq={seq}  (showing only variants ≤2 AA away)"],
        idx,
    )


@app.callback(
    Output("download-csv", "data"),
    Output("export-status", "children"),
    Input("export-button", "n_clicks"),
    State("selected-idx-store", "data"),
    prevent_initial_call=True,
)
def export_sequences(n_clicks, selected_idx):
    """Export highlighted sequences to CSV."""
    if selected_idx is None:
        return no_update, "No sequence selected. Click on a point first."

    selected_seq = df.loc[selected_idx, "aa_sequence"]
    dists = df["aa_sequence"].apply(lambda s: hamming(selected_seq, s)).to_numpy()
    highlighted_indices = np.where(np.isin(dists, [0, 1, 2]))[0]

    export_data = [df.iloc[selected_idx]]
    for idx in highlighted_indices:
        if idx != selected_idx:
            export_data.append(df.iloc[idx])

    export_df = pd.DataFrame(export_data).reset_index(drop=True)

    variant_id = df.loc[selected_idx, "variant_id"]
    filename = (
        str(variant_id)
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(", ", "_")
    )
    filename = f"{filename}.csv"

    return dict(
        content=export_df.to_csv(index=False), filename=filename
    ), f"Exported {len(export_df)} sequences to {filename}"


def main():
    """Main function to run the interactive scatter plot app."""
    global df, base_colors
    parser = build_parser()
    args = parser.parse_args()

    # Load and validate CSV
    required_columns = {"aa_sequence", "log10_rep1_Kd", "log10_rep2_Kd", "variant_id"}
    try:
        df = pd.read_csv(args.i).reset_index(drop=True)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        print(
            f"Error: CSV is missing required columns: {', '.join(sorted(missing_cols))}"
        )
        sys.exit(1)

    print(f"Loaded {len(df)} sequences from {args.i}")

    # Run the Dash app
    app.run(debug=False)


if __name__ == "__main__":
    main()
