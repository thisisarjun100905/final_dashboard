import pandas as pd
from typing import Iterable, Optional, Union
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
    """
    Ensure a boolean Series mask aligned to index, with NaNs -> False.
    Accepts array-like / series / scalar.
    """
    m = _as_series(mask_like, index=index)
    m = m.fillna(False)
    if m.dtype != bool:
        m = m.astype(bool)
    return m


# ---------------------------
# Data prep
# ---------------------------
def filter_data(
    df: pd.DataFrame,
    day: Union[int, float],
    campaign: Optional[Union[int, Iterable[int]]] = None,
    target_col: str = "",
    flag: bool = True,  # True: == day, False: <= day
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [month, branch, target_col]
    aggregated (sum) by (month, branch) with a day filter and optional campaign filter.

    - flag=True  -> day == given day
    - flag=False -> day <= given day
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "branch", target_col])

    idx = df.index

    # Day filter
    day_col = pd.to_numeric(df.get("day"), errors="coerce")
    day_mask = _safe_bool_mask((day_col == day) if flag else (day_col <= day), idx)

    # Campaign filter (optional)
    if campaign is not None:
        if not isinstance(campaign, Iterable) or isinstance(campaign, (str, bytes, int)):
            campaign = [campaign]
        camp_series = pd.to_numeric(df.get("campaign_id"), errors="coerce")
        camp_mask = _safe_bool_mask(camp_series.isin(list(campaign)), idx)
        mask = _safe_bool_mask(day_mask & camp_mask, idx)
    else:
        mask = day_mask

    # Ensure numeric target (create zeros if missing)
    if target_col in df.columns:
        tgt_raw = df[target_col]
    else:
        tgt_raw = pd.Series([0] * len(df), index=df.index)
    tgt = pd.to_numeric(tgt_raw, errors="coerce").fillna(0)

    # Build and group
    month_series = pd.to_numeric(df.get("month"), errors="coerce")
    branch_series = _as_series(df.get("branch"), index=idx, dtype="string")

    grouped = (
        pd.DataFrame({"month": month_series, "branch": branch_series, target_col: tgt})
        .loc[mask]
        .dropna(subset=["month", "branch"])
        .groupby(["month", "branch"], observed=True, as_index=False)[target_col]
        .sum()
    )
    return grouped


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
        x="branch",
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
        xaxis_title="Branch",
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
def filter_data_conv(df: pd.DataFrame, campaign, day) -> pd.DataFrame:
    """
    Filters by 'day' and campaign(s), groups by (branch, month),
    sums hot_team_leads & unique_leads, and computes conversion.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "branch", "month", "hot_team_leads", "unique_leads", "conversion"
        ])

    d = df.copy()

    # Day filter
    d_day = pd.to_numeric(d.get("day"), errors="coerce")
    d = d[d_day == day]

    # Campaign filter (scalar or iterable)
    if isinstance(campaign, Iterable) and not isinstance(campaign, (str, bytes)):
        d = d[d["campaign_id"].isin(list(campaign))]
    else:
        d = d[d["campaign_id"] == campaign]

    if d.empty:
        return pd.DataFrame(columns=[
            "branch", "month", "hot_team_leads", "unique_leads", "conversion"
        ])

    grouped = (
        d.groupby(["branch", "month"], dropna=False)
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
    return grouped[["branch", "month", "hot_team_leads", "unique_leads", "conversion"]]

def pan_india(df: pd.DataFrame, target_col: str, flag: bool) -> pd.DataFrame:
    """
    If flag=True  -> return [% conversion] by month x pan_ind_{main,remote}
    If flag=False -> return [sum(target_col)] by month x pan_ind_{main,remote}
    """
    # --- Guards for empty / missing essentials ---
    if df is None or df.empty or "branch" not in df.columns or "month" not in df.columns:
        if flag:
            return pd.DataFrame(columns=["month", "branch", "conversion"])
        else:
            return pd.DataFrame(columns=["month", "branch", target_col])

    d = df.copy()

    # branch_flag: 1 for *_main, 0 otherwise (vectorized)
    branch_suffix = d["branch"].astype(str).str.rsplit("_", n=1).str[-1]
    d["branch_flag"] = (branch_suffix == "main").astype(int)

    def _ensure_numeric_col(df_: pd.DataFrame, col: str) -> None:
        """Ensure df_[col] exists and is numeric with NaNs -> 0."""
        if col not in df_.columns:
            df_[col] = 0.0
        df_[col] = pd.to_numeric(df_[col], errors="coerce").fillna(0)

    if flag:
        # Need both hot_team_leads & unique_leads to compute conversion%
        _ensure_numeric_col(d, "hot_team_leads")
        _ensure_numeric_col(d, "unique_leads")

        grp = (
            d.groupby(["month", "branch_flag"], dropna=False)
             .agg({"hot_team_leads": "sum", "unique_leads": "sum"})
             .reset_index()
        )

        # Safe division: zeros -> 0%
        denom = grp["unique_leads"].replace(0, np.nan)
        grp["conversion"] = (grp["hot_team_leads"] * 100.0 / denom).fillna(0).round(2)

        grp["branch"] = "pan_ind" + grp["branch_flag"].map({1: "_main", 0: "_remote"})
        out = grp[["month", "branch", "conversion"]]
        return out

    else:
        # Sum the requested target_col
        _ensure_numeric_col(d, target_col)

        grp = (
            d.groupby(["month", "branch_flag"], dropna=False)
             .agg({target_col: "sum"})
             .reset_index()
        )
        grp["branch"] = "pan_ind" + grp["branch_flag"].map({1: "_main", 0: "_remote"})
        out = grp[["month", "branch", target_col]]
        return out
    


def pan_india_rate(df: pd.DataFrame, num_col: str, deno_col: str ,target_col : str) -> pd.DataFrame:
    d = df.copy()
    # branch_flag: 1 for *_main, 0 otherwise (vectorized)
    branch_suffix = d["branch"].astype(str).str.rsplit("_", n=1).str[-1]
    d["branch_flag"] = (branch_suffix == "main").astype(int)
    grp = (
            d.groupby(["month", "branch_flag"], dropna=False)
             .agg({num_col: "sum", deno_col: "sum"})
             .reset_index()
        )
    denom = grp[deno_col].replace(0, np.nan)
    grp[target_col] = (grp[num_col] * 100.0 / denom).fillna(0).round(2)

    grp["branch"] = "pan_ind" + grp["branch_flag"].map({1: "_main", 0: "_remote"})
    out = grp[["month", "branch", target_col]]
    return out


def filter_appointment_rate_branch(df_app , df_req , col_num , col_deno , col_tar ,  flag):
    df_app_rate = df_app.merge(
    df_req,
    on=['month', 'branch'],
    how='inner'
    )
    num = pd.to_numeric(df_app_rate[col_num], errors='coerce').fillna(0)
    den = pd.to_numeric(df_app_rate[col_deno], errors='coerce').fillna(0)
    safe_den = den.replace(0, np.nan)
    if flag:
        df_app_rate[col_tar] = (num / safe_den * 100).round(2).fillna(0)
    else:
        safe_num = num.replace(0, np.nan)
        df_app_rate[col_tar] = (den / safe_num * 100).round(2).fillna(0)
    return df_app_rate

