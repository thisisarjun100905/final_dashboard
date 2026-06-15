import pandas as pd
import plotly.graph_objects as go


def format_inr(value, decimals: int = 0) -> str:
    """Format a number using the Indian numbering system.

    e.g. 21288102 -> '2,12,88,102'  (last 3 digits, then groups of 2).
    `decimals` controls fractional digits (0 for whole rupees).
    """
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(n):
        return ""
    neg = n < 0
    txt = f"{abs(n):.{decimals}f}"            # full number, rounded — avoids carry bugs
    if "." in txt:
        int_str, frac = txt.split(".")
        frac = "." + frac
    else:
        int_str, frac = txt, ""
    if len(int_str) > 3:
        last3 = int_str[-3:]
        rest = int_str[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        int_str = ",".join(parts) + "," + last3
    return ("-" if neg else "") + int_str + frac


def hover_kwargs(yvals, label, value_type: str = "number", decimals: int = 0) -> dict:
    """Build customdata + hovertemplate so hover shows the value in the main box
    and "<day> <label>" (e.g. "14 Jun") in the colored side box.

      "rupee"   -> Indian-formatted value (via customdata)
      "percent" -> "%{y:.2f}%"
      "number"  -> "%{y:,.0f}"
    """
    side = "<extra>%{x} " + str(label) + "</extra>"
    if value_type == "rupee":
        return dict(
            customdata=[format_inr(v, decimals) for v in yvals],
            hovertemplate="%{customdata}" + side,
        )
    if value_type == "percent":
        return dict(hovertemplate="%{y:.2f}%" + side)
    return dict(hovertemplate="%{y:,.0f}" + side)


def make_timeseries_figure(
    df: pd.DataFrame,
    months: list[int],
    ycol: str,
    title: str,
    ytitle: str,
    value_type: str = "number"
) -> go.Figure:
    """Build a multi-month line chart for the given ycol.

    value_type controls hover + axis formatting:
      "rupee"   -> "(Rs)" axis label, comma-grouped values (no "M"/Millions
                   abbreviation) on both ticks and hover.
      "percent" -> "(%)" axis label, values shown with a "%" suffix and 2
                   decimals on hover.
      "number"  -> left untouched (plain counts like leads/appointments/deals).
    """

    fig = go.Figure()

    if df is None or df.empty or ycol not in df.columns:
        fig.update_layout(
            annotations=[dict(
                text=f"No data for '{ycol}'.",
                x=0.5,
                y=0.5,
                showarrow=False
            )],
            height=420,
            margin=dict(l=50, r=30, t=60, b=40),
            showlegend=True
        )
        return fig

    MONTH_NAME = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    color_palette = [
        "#4983f6",  # orange
        "#38cb38",  # green
        "#fd3a3a",  # red
        "#e2a60e",  # purple
        "#e377c2",  # pink
        "#bcbd22",  # olive
        "#8c564b",  # brown
    ]

    # Multi-month plot
    if 'month' in df.columns and months:
        for idx, m in enumerate(months):
            mdf = df[df['month'] == m]
            if not mdf.empty:
                color = color_palette[idx % len(color_palette)]

                month_label = MONTH_NAME.get(m, f"Month {m}")
                fig.add_trace(go.Scatter(
                    x=mdf['day'] if 'day' in mdf.columns else mdf.index,
                    y=mdf[ycol],
                    mode='lines+markers',
                    name=month_label,
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=6),
                    **hover_kwargs(mdf[ycol], month_label, value_type)
                ))
    else:
        fig.add_trace(go.Scatter(
            x=df['day'] if 'day' in df.columns else df.index,
            y=df[ycol],
            mode='lines+markers',
            name=title,
            line=dict(color=color_palette[0], width=2),
            marker=dict(color=color_palette[0], size=6),
            **hover_kwargs(df[ycol], title, value_type)
        ))

    # ---- Hover / axis formatting by value type ----
    yaxis_opts = dict(title=ytitle)
    if value_type == "rupee":
        yaxis_opts["title"] = f"{ytitle} (Rs)"
        yaxis_opts["tickformat"] = ",.0f"   # commas, no "M"/Millions abbreviation
        yaxis_opts["hoverformat"] = ",.0f"
    elif value_type == "percent":
        yaxis_opts["title"] = f"{ytitle} (%)"
        yaxis_opts["ticksuffix"] = "%"      # appended to hover values too
        yaxis_opts["hoverformat"] = ".2f"

    fig.update_layout(
        xaxis_title="Day of Month" if 'day' in df.columns else "Index",
        yaxis=yaxis_opts,

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,            # ? legend at bottom
            xanchor="center",
            x=0.5
        ),

        showlegend=True,
        margin=dict(l=50, r=30, t=60, b=90),
        height=420
    )

    return fig

def make_monthly_line_bar(
    df: pd.DataFrame,
    months: list[int],
    bar_col: str,
    line_col: str,
    title: str
) -> go.Figure:

    fig = go.Figure()

    if df is None or df.empty:
        fig.update_layout(
            annotations=[dict(
                text="No data available.",
                x=0.5,
                y=0.5,
                showarrow=False
            )]
        )
        return fig

    MONTH_NAME = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    color_palette = [
        "#4983f6",
        "#38cb38",
        "#fd3a3a",
        "#e2a60e",
        "#e377c2",
        "#8c564b",
    ]

    for idx, m in enumerate(months):

        mdf = df[df['month'] == m]

        if mdf.empty:
            continue

        color = color_palette[idx % len(color_palette)]
        month_label = MONTH_NAME.get(m, f"Month {m}")

        # 🔹 Bar (no_emp)
        fig.add_trace(go.Bar(
            x=mdf['day'],
            y=mdf[bar_col],
            name=f"{month_label} - {bar_col}",
            marker=dict(color=color),
            legendgroup=month_label
        ))

        # 🔹 Line (avg_tme_experience)
        fig.add_trace(go.Scatter(
            x=mdf['day'],
            y=mdf[line_col],
            mode='lines+markers',
            name=f"{month_label} - {line_col}",
            line=dict(color=color, width=3),
            marker=dict(size=6),
            yaxis='y2',
            legendgroup=month_label
        ))

    fig.update_layout(
        title=title,

        xaxis=dict(title="Day"),

        yaxis=dict(
            title=bar_col
        ),

        yaxis2=dict(
            title=line_col,
            overlaying='y',
            side='right'
        ),

        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),

        barmode='group',
        height=450,
        template="plotly_white"
    )

    return fig