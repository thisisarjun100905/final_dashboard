import pandas as pd
from typing import Iterable, Optional, Union, List
import numpy as np


# ---------------------------
# Helpers
# ---------------------------
def _as_series(x, index, dtype=None, default=None) -> pd.Series:
    """Coerce x to a pandas Series aligned to index."""
    if isinstance(x, pd.Series):
        s = x.reindex(index)
    else:
        s = pd.Series(x, index=index)

    if dtype is not None:
        try:
            s = s.astype(dtype)
        except Exception:
            pass

    if default is not None:
        s = s.fillna(default)

    return s


def _safe_bool_mask(mask_like, index) -> pd.Series:
    """Ensure a boolean Series mask aligned to index, with NaNs -> False."""
    m = _as_series(mask_like, index=index)
    m = m.fillna(False)

    if m.dtype != bool:
        m = m.astype(bool)

    return m


def filter_data(
    df: pd.DataFrame,
    day: Union[int, float],
    branch: Optional[List[str]] = None,
    target_col: str = "",
    flag: bool = True,
) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "campaign_name", target_col])

    idx = df.index

    # --------------------
    # Day filter
    # --------------------
    day_col = pd.to_numeric(df.get("day"), errors="coerce")
    day_mask = _safe_bool_mask(
        (day_col == day) if flag else (day_col <= day),
        idx
    )

    # --------------------
    # Campaign filter
    # --------------------
    if branch is not None:
        if not isinstance(branch, Iterable) or isinstance(branch, (str, bytes, int)):
            branch = [branch]

        branch_series = _as_series(
            df.get("branch"),
            index=idx,
            dtype="string"
        )

        branch_mask = _safe_bool_mask(branch_series.isin(branch), idx)
        mask = day_mask & branch_mask
    else:
        mask = day_mask

    # --------------------
    # Target column
    # --------------------
    if target_col in df.columns:
        tgt = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    else:
        tgt = pd.Series(0, index=idx)

    # --------------------
    # Grouping
    # --------------------
    month_series = pd.to_numeric(df.get("month"), errors="coerce")
    camp_series = _as_series(df.get("campaign_name"), index=idx, dtype="string")

    result = (
        pd.DataFrame(
            {
                "month": month_series,
                "campaign_name": camp_series,
                target_col: tgt,
            }
        )
        .loc[mask]
        .dropna(subset=["month", "campaign_name"])
        .groupby(
            ["month", "campaign_name"],
            as_index=False,
            observed=True
        )[target_col]
        .sum()
    )

    return result

# ---------------------------
# Plotting
# ---------------------------
def plot_grouped_bar(group_df: pd.DataFrame, target_col: str, graph_title: str):
    """
    Grouped bar chart (x=branch, y=target_col, color=month).
    - Last 4 months only (calendar correct: DEC → JAN)
    - Y-axis auto-scaled (whole numbers if large)
    - Hover shows only target value rounded to 2 decimals
    """

    import plotly.express as px
    import pandas as pd

    MONTH_MAP = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR",
        5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG",
        9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
    }

    if group_df is None or group_df.empty:
        return px.bar(title=f"{graph_title} — No data")

    df = group_df.copy()

    # Normalize month
    df["month_num"] = pd.to_numeric(df["month"], errors="coerce")

    # Handle year for DEC → JAN
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        current_month = pd.Timestamp.today().month
        current_year = pd.Timestamp.today().year
        df["year"] = df["month_num"].apply(
            lambda m: current_year - 1 if m > current_month else current_year
        )

    # Create datetime for correct ordering
    df["month_date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month_num"], day=1)
    )

    # Pick last 4 months
    last_4_dates = (
        df["month_date"]
        .dropna()
        .sort_values()
        .unique()[-4:]
    )

    df = df[df["month_date"].isin(last_4_dates)]

    # Ordered months
    ordered_months = (
        df.sort_values("month_date")[["month_date", "month_num"]]
        .drop_duplicates()["month_num"]
        .tolist()
    )

    month_labels = [MONTH_MAP[m] for m in ordered_months]

    df["month"] = pd.Categorical(
        df["month_num"].map(MONTH_MAP),
        categories=month_labels,
        ordered=True
    )

    fig = px.bar(
        df,
        x="campaign_name",
        y=target_col,
        color="month",
        barmode="group",
        title=graph_title,
        category_orders={"month": month_labels},
        hover_data={target_col: False}  # remove default hover
    )

    # Hover: only target value (2 decimals)
    fig.update_traces(
        hovertemplate=f"{target_col}: %{{y:.2f}}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="campaign_name",
        yaxis_title=target_col.replace("_", " ").title(),
        legend_title="Month",
        bargap=0.25,
        width=1000,
        height=450,
        margin=dict(l=40, r=20, t=60, b=100),
        yaxis=dict(
            tickformat=",",      # whole numbers when large
            separatethousands=True
        )
    )

    fig.update_xaxes(tickangle=-35, automargin=True)

    return fig


# ---------------------------
# Conversion aggregation
# ---------------------------
def filter_data_conv(df: pd.DataFrame, day , branch : Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filters by 'day' and campaign(s), groups by (branch, month),
    sums hot_team_leads & unique_leads, and computes conversion.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "campaign_name", "month", "hot_team_leads", "unique_leads", "conversion"
        ])

    d = df.copy()

    # Day filter
    d_day = pd.to_numeric(d.get("day"), errors="coerce")
    d = d[d_day == day]

    # Campaign filter (scalar or iterable)
    if isinstance(branch, Iterable) and not isinstance(branch, (str, bytes)):
        d = d[d["branch"].isin(list(branch))]
    else:
        d = d[d["branch"] == branch]

    if d.empty:
        return pd.DataFrame(columns=[
            "campaign_name", "month", "hot_team_leads", "unique_leads", "conversion"
        ])

    grouped = (
        d.groupby(["campaign_name", "month"], dropna=False)
         .agg({
             "hot_team_leads": "sum",
             "unique_leads": "sum"
         })
         .reset_index()
    )

    # Safe division
    grouped["hot_team_leads"] = pd.to_numeric(grouped["hot_team_leads"], errors="coerce").fillna(0)
    grouped["unique_leads"] = pd.to_numeric(grouped["unique_leads"], errors="coerce").fillna(0)

    denom = grouped["unique_leads"].replace(0, pd.NA)
    grouped["conversion"] = (grouped["hot_team_leads"] / denom).fillna(0)
    grouped['conversion']=  (grouped['conversion']*100).round(2)
    return grouped[["campaign_name", "month", "hot_team_leads", "unique_leads", "conversion"]]


def filter_appointment_rate_branch(
    df_app,
    df_req,
    col_num,
    col_deno,
    col_tar,
    flag
    ):

    df_app_rate = df_app.merge(
        df_req,
        on=["month", "campaign_name"],
        how="inner"
    )

    num = pd.to_numeric(df_app_rate[col_num], errors="coerce")
    den = pd.to_numeric(df_app_rate[col_deno], errors="coerce")

    # Default rate = 0
    rate = np.zeros(len(df_app_rate))

    # Valid condition:
    # 1) denominator > 0
    # 2) numerator <= denominator
    valid_mask = (den > 0) & (num <= den)

    rate[valid_mask] = (num[valid_mask] / den[valid_mask] * 100).round(2)

    df_app_rate[col_tar] = rate

    return df_app_rate
