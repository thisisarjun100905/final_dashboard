import pandas as pd
import numpy as np
from typing import List, Optional

# ---------------------------
# Internal helpers
# ---------------------------

def _rename_std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soft-standardize common column names:
      - month_no -> month
      - no_days  -> day
    """
    df = df.copy()
    col_map = {}
    if "month_no" in df.columns and "month" not in df.columns:
        col_map["month_no"] = "month"
    if "no_days" in df.columns and "day" not in df.columns:
        col_map["no_days"] = "day"
    if col_map:
        df = df.rename(columns=col_map)
    return df

def _to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float")

def _norm_str_series(s: pd.Series) -> pd.Series:
    """Normalize strings for robust string comparisons."""
    return (
        s.astype(str)
         .str.replace(r"[\r\n\t]+", " ", regex=True)
         .str.strip()
         .str.lower()
    )

def _apply_branch_filter(df: pd.DataFrame, branches: Optional[List[str]]) -> pd.DataFrame:
    """Filter by branch if provided (case/whitespace-insensitive)."""
    if not branches or "branch" not in df.columns:
        return df
    target = {str(b).strip().lower() for b in branches if isinstance(b, str) and b.strip()}
    if not target:
        return df
    cur = _norm_str_series(df["branch"])
    return df[cur.isin(target)]

def _coerce_and_filter(
    df: Optional[pd.DataFrame],
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Standardize, coerce types, and filter by campaign_ids, months, and optional branches.
    Drops rows where month/day are missing (when both columns exist).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = _rename_std_cols(df)

    # Coerce key columns if they exist
    if "campaign_id" in out.columns:
        out["campaign_id"] = _to_int_series(out["campaign_id"])
    if "month" in out.columns:
        out["month"] = _to_int_series(out["month"])
    if "day" in out.columns:
        out["day"] = _to_int_series(out["day"])

    # Filter by campaign ids (empty list = no filter, mirrors month/branch behavior)
    if "campaign_id" in out.columns and campaign_ids:
        out = out[out["campaign_id"].isin(pd.Series(campaign_ids, dtype="Int64"))]
    if "month" in out.columns and months:
        out = out[out["month"].isin(pd.Series(months, dtype="Int64"))]

    # Branch filter (optional)
    out = _apply_branch_filter(out, branches)

    # Drop missing month/day only if both columns exist
    if {"month", "day"}.issubset(out.columns):
        out = out.dropna(subset=["month", "day"])

    return out

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ---------------------------
# Public filters
# ---------------------------


def filter_appointment_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days, appointment_close|appointments|appointment, branch?
    Output columns:
      month, day, appointment_close (sum), appointment (cumulative within month)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "appointment_close", "appointment"])

    app_col = _first_existing_col(base, ["appointment_close", "appointments", "appointment"])
    if app_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "appointment_close", "appointment"])

    base[app_col] = _to_float_series(base[app_col]).fillna(0)
    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(appointment_close=(app_col, "sum"))
            .sort_values(["month", "day"])
    )
    agg["appointment"] = agg.groupby("month", group_keys=False)["appointment_close"].cumsum()
    return agg

def filter_lead_generation_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days, Leads_created|leads_created|leads, branch?
    Output columns:
      month, day, Leads_created (sum), Leads_created_cum (cumulative within month)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "Leads_created", "Leads_created_cum"])

    leads_col = _first_existing_col(base, ["Leads_created", "leads_created", "leads"])
    if leads_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "Leads_created", "Leads_created_cum"])

    base[leads_col] = _to_float_series(base[leads_col]).fillna(0)

    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(Leads_created=(leads_col, "sum"))
            .sort_values(["month", "day"])
    )
    agg["Leads_created_cum"] = agg.groupby("month", group_keys=False)["Leads_created"].cumsum()
    return agg

def filter_conversion_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days,
      hot_team_leads|hot_leads, unique_leads|unique, branch?
    Output columns:
      month, day, hot_team_leads (sum), unique_leads (sum), conversion (%)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "hot_team_leads", "unique_leads", "conversion"])

    hot_col  = _first_existing_col(base, ["hot_team_leads", "hot_leads"])
    uniq_col = _first_existing_col(base, ["unique_leads", "unique"])
    rv_3_yr =  _first_existing_col(base , ['rv_3_yr_hot_team' , 'rv_3_yr'])
    if (hot_col is None or uniq_col is None) or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "hot_team_leads", "unique_leads", "conversion" , "rv_3_yr_hot_team"])

    base[hot_col]  = _to_float_series(base[hot_col]).fillna(0)
    base[uniq_col] = _to_float_series(base[uniq_col]).fillna(0)

    agg_dict = {
        "hot_team_leads": (hot_col, "sum"),
        "unique_leads": (uniq_col, "sum"),
    }
    if rv_3_yr is not None:
        base[rv_3_yr] = _to_float_series(base[rv_3_yr]).fillna(0)
        agg_dict["rv_3_yr_hot_team"] = (rv_3_yr, "sum")

    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(**agg_dict)
            .sort_values(["month", "day"])
    )
    if "rv_3_yr_hot_team" not in agg.columns:
        agg["rv_3_yr_hot_team"] = 0
    denom = agg["unique_leads"].replace(0, np.nan)
    agg["conversion"] = pd.Series(np.where(
    agg["hot_team_leads"] > denom,
    0,
    ((agg["hot_team_leads"] / denom) * 100)
    ), index=agg.index).round(2).fillna(0)
    return agg


def filter_NIB_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days, not_in_buss|NIB
    Output columns:
      month, day, NIB (sum), NIB_CUM
    """
    
    # Step 1 — Filter and standardize dataframe
    base = _coerce_and_filter(df, campaign_ids, months, branches)

    if base.empty:
        return pd.DataFrame(columns=["month", "day", "NIB", "NIB_CUM"])

    # Step 2 — Find the NIB column (flexible naming)
    nib_col = _first_existing_col(
        base,
        ["not_in_buss", "NIB", "not_in_business"]
    )

    # If required columns missing → return empty safe df
    if nib_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "NIB", "NIB_CUM"])

    # Step 3 — Ensure nib_col is numeric
    base[nib_col] = pd.to_numeric(base[nib_col], errors='coerce').fillna(0)

    # Step 4 — Group and aggregate
    agg = (
        base
        .groupby(["month", "day"], as_index=False)
        .agg(NIB=(nib_col, "sum"))
        .sort_values(["month", "day"])
    )

    # Step 5 — Cumulative NIB within each month
    agg["NIB_CUM"] = agg.groupby("month", group_keys=False)["NIB"].cumsum()

    # Step 6 — Final cleanup for safety
    agg["NIB"] = agg["NIB"].fillna(0)
    agg["NIB_CUM"] = agg["NIB_CUM"].fillna(0)
    return agg


def filter_nib_rate(df_nib, df_leads, columns, col_name):
    # Step 1 — Validate required columns exist
    required_cols = {columns, "month", "day"}
    if not required_cols.issubset(df_leads.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    if not {"month", "day", "NIB_CUM"}.issubset(df_nib.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    # Step 2 — Keep only required columns from leads df
    df_leads = df_leads[[columns, "month", "day"]]

    # Step 3 — Merge
    df_nib_rate = df_leads.merge(
        df_nib,
        on=["month", "day"],
        how="inner"
    )

    # Step 4 — Convert numeric columns safely
    df_nib_rate["NIB_CUM"] = pd.to_numeric(df_nib_rate["NIB_CUM"], errors="coerce").fillna(0)
    df_nib_rate[columns] = pd.to_numeric(df_nib_rate[columns], errors="coerce").fillna(0)

    # Step 5 — Replace zeros in denominator to prevent divide-by-zero errors
    df_nib_rate[columns] = df_nib_rate[columns].replace(0, np.nan)

    # Step 6 — Calculate NIB rate
    df_nib_rate[col_name] = pd.Series(np.where(
        df_nib_rate["NIB_CUM"] > df_nib_rate[columns],
        0,
        (df_nib_rate["NIB_CUM"] / df_nib_rate[columns]) * 100
    ), index=df_nib_rate.index).round(2).fillna(0)
        # Step 7 — Clean NaN results
    df_nib_rate[col_name] = df_nib_rate[col_name].fillna(0)

    return df_nib_rate


def filter_appointment_rate(df_app, df_req, columns, col_name, flag):
    # Merge both dataframes
    df_app_rate = df_app.merge(
        df_req,
        on=['month', 'day'],
        how='left'
    )

    # Convert required columns to numeric (important)
    df_app_rate['appointment'] = pd.to_numeric(df_app_rate['appointment'], errors='coerce')
    df_app_rate[columns] = pd.to_numeric(df_app_rate[columns], errors='coerce')
    df_app_rate['appointment'] = df_app_rate['appointment'].fillna(
    df_app_rate.groupby('month')['appointment'].transform('max')
    )
    df_app_rate[columns] = df_app_rate[columns].fillna(
        df_app_rate.groupby('month')[columns].transform('max')
    )

    if flag:
        # appointment / columns
        df_app_rate[col_name] = pd.Series(np.where(
            df_app_rate['appointment'] >= df_app_rate[columns],
            0,
            (df_app_rate['appointment'] / df_app_rate[columns]) * 100
        ), index=df_app_rate.index).round(2).fillna(0)
    else:
        # columns / appointment
        df_app_rate[col_name] = pd.Series(np.where(
            df_app_rate[columns] >= df_app_rate['appointment'],
            0,
            (df_app_rate[columns] / df_app_rate['appointment']) * 100
        ), index=df_app_rate.index).round(2).fillna(0)

    # Final cleanup: Replace NaN back to 0 for display
    df_app_rate[col_name] = df_app_rate[col_name].fillna(0)
    return df_app_rate



def filter_ni_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days, appointment_close|appointments|appointment, branch?
    Output columns:
      month, day, appointment_close (sum), appointment (cumulative within month)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "NI", "NI_CUM"])

    app_col = _first_existing_col(base, ["not_interested"])
    if app_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "NI", "NI_CUM"])


    base[app_col] = _to_float_series(base[app_col]).fillna(0)

    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(NI=(app_col, "sum"))
            .sort_values(["month", "day"])
    )
    agg["NI_CUM"] = agg.groupby("month", group_keys=False)["NI"].cumsum()
    return agg

def filter_ni_rate(df_ni, df_leads, columns, col_name):
    # Step 1 — Validate required columns exist in df_leads
    required_cols_leads = {columns, "month", "day"}
    if not required_cols_leads.issubset(df_leads.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    # Step 2 — Validate required NI columns exist in df_ni
    if not {"month", "day", "NI_CUM"}.issubset(df_ni.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    # Step 3 — Keep necessary columns from leads
    df_leads = df_leads[[columns, "month", "day"]]

    # Step 4 — Merge
    df_ni_rate = df_leads.merge(
        df_ni,
        on=["month", "day"],
        how="inner"
    )

    # Step 5 — Convert numeric columns safely
    df_ni_rate["NI_CUM"] = pd.to_numeric(df_ni_rate["NI_CUM"], errors="coerce").fillna(0)
    df_ni_rate[columns] = pd.to_numeric(df_ni_rate[columns], errors="coerce").fillna(0)

    # Step 6 — Avoid division by zero
    df_ni_rate[columns] = df_ni_rate[columns].replace(0, np.nan)

    # Step 7 — Calculate NI rate
    df_ni_rate[col_name] = pd.Series(np.where(
        df_ni_rate["NI_CUM"] > df_ni_rate[columns],
        0,
        (df_ni_rate["NI_CUM"] / df_ni_rate[columns]) * 100
    ), index=df_ni_rate.index).round(2).fillna(0)
    # Step 8 — Clean NaN in final output
    df_ni_rate[col_name] = df_ni_rate[col_name].fillna(0)

    return df_ni_rate



def filter_attempt_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days, appointment_close|appointments|appointment, branch?
    Output columns:
      month, day, appointment_close (sum), appointment (cumulative within month)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "attemp_count"])

    app_col = _first_existing_col(base, ["attempt_count"])
    leads_col = _first_existing_col(base, ["leads"])
    if app_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "attempt_count" , "leads"])

    base[app_col] = _to_float_series(base[app_col]).fillna(0)
    base[leads_col] = _to_float_series(base[leads_col]).fillna(0)
    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(attempt_cum=(app_col, "sum") ,
                 leads_cum = (leads_col , "sum"))
            .sort_values(["month", "day"])
    )
    return agg


def filter_attempt_rate(df_attempt, col_name):

    # Step 2 — Validate df_attempt required columns
    if not {"month", "day", "attempt_cum" , "leads_cum"}.issubset(df_attempt.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    df_attempt_rate = df_attempt

    # Step 5 — Convert numeric fields safely
    df_attempt_rate["attempt_cum"] = pd.to_numeric(df_attempt_rate["attempt_cum"], errors="coerce").fillna(0)
    df_attempt_rate["leads_cum"] = pd.to_numeric(df_attempt_rate["leads_cum"], errors="coerce").fillna(0)

    # Step 6 — Handle denominator zero to avoid division-by-zero
    den = df_attempt_rate["leads_cum"].replace(0, np.nan)

    # Step 7 — Calculate attempt rate
    df_attempt_rate[col_name] = pd.Series(np.where(
        df_attempt_rate["attempt_cum"] > den,
        0,
        (df_attempt_rate["attempt_cum"] / den) * 100
    ), index=df_attempt_rate.index).round(2).fillna(0)

    # Step 8 — Final cleanup: replace NaN with 0
    df_attempt_rate[col_name] = df_attempt_rate[col_name].fillna(0)

    return df_attempt_rate


# RV to Leads Rate = (rv_3_year_hot_team / unique_leads) * 100
def filter_rv_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns (flexible):
      campaign_id, month|month_no, day|no_days, appointment_close|appointments|appointment, branch?
    Output columns:
      month, day, appointment_close (sum), appointment (cumulative within month)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "unique_leads" , "rv_3_yr_hot_team"])

    app_col = _first_existing_col(base, ["rv_3_yr_hot_team", "rv_3_yr"])
    leads_col = _first_existing_col(base, ["unique_leads", "unique"])
    if app_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "unique_leads" , "rv_3_yr_hot_team"])

    base[app_col] = _to_float_series(base[app_col]).fillna(0)
    base[leads_col] = _to_float_series(base[leads_col]).fillna(0)
    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(rv_3_yr_hot_team_cum=(app_col, "sum") ,
                 unique_leads_cum = (leads_col , "sum"))
            .sort_values(["month", "day"])
    )
    return agg

def filter_rv_rate(df_rv, col_name):

    # Step 2 — Validate df_rv required columns
    if not {"month", "day", "rv_3_yr_hot_team_cum" , "unique_leads_cum"}.issubset(df_rv.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    df_rv_rate = df_rv

    # Step 5 — Convert numeric fields safely
    df_rv_rate["rv_3_yr_hot_team_cum"] = pd.to_numeric(df_rv_rate["rv_3_yr_hot_team_cum"], errors="coerce").fillna(0)
    df_rv_rate["unique_leads_cum"] = pd.to_numeric(df_rv_rate["unique_leads_cum"], errors="coerce").fillna(0)

    # Step 6 — Handle denominator zero to avoid division-by-zero
    den = df_rv_rate["unique_leads_cum"].replace(0, np.nan)

    # Step 7 — Calculate rv rate
    df_rv_rate[col_name] = ((df_rv_rate["rv_3_yr_hot_team_cum"] / den)).round(2).fillna(0)

    # Step 8 — Final cleanup: replace NaN with 0
    df_rv_rate[col_name] = df_rv_rate[col_name].fillna(0)

    return df_rv_rate




def filter_bad_dispo_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns:
      campaign_id, month, day, bad_dispo, branch?
    Output columns:
      month, day, BAD_DISPO (sum), BAD_DISPO_CUM (cumulative within month)
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "BAD_DISPO", "BAD_DISPO_CUM"])

    col = _first_existing_col(base, ["bad_dispo", "Bad_Dispo", "bad_disposition"])
    if col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "BAD_DISPO", "BAD_DISPO_CUM"])

    base[col] = _to_float_series(base[col]).fillna(0)

    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(BAD_DISPO=(col, "sum"))
            .sort_values(["month", "day"])
    )
    agg["BAD_DISPO_CUM"] = agg.groupby("month", group_keys=False)["BAD_DISPO"].cumsum()
    agg["BAD_DISPO"] = agg["BAD_DISPO"].fillna(0)
    agg["BAD_DISPO_CUM"] = agg["BAD_DISPO_CUM"].fillna(0)
    return agg


def filter_bad_dispo_rate(df_bad_dispo, df_leads, columns, col_name):
    """
    Bad Disposition Rate = (BAD_DISPO_CUM / Leads_created_cum) * 100
    Same logic as NI rate.
    """
    required_cols_leads = {columns, "month", "day"}
    if not required_cols_leads.issubset(df_leads.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    if not {"month", "day", "BAD_DISPO_CUM"}.issubset(df_bad_dispo.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    df_leads = df_leads[[columns, "month", "day"]]

    df_rate = df_leads.merge(
        df_bad_dispo,
        on=["month", "day"],
        how="inner"
    )

    df_rate["BAD_DISPO_CUM"] = pd.to_numeric(df_rate["BAD_DISPO_CUM"], errors="coerce").fillna(0)
    df_rate[columns] = pd.to_numeric(df_rate[columns], errors="coerce").fillna(0)

    df_rate[columns] = df_rate[columns].replace(0, np.nan)

    df_rate[col_name] = pd.Series(np.where(
        df_rate["BAD_DISPO_CUM"] > df_rate[columns],
        0,
        (df_rate["BAD_DISPO_CUM"] / df_rate[columns]) * 100
    ), index=df_rate.index).round(2).fillna(0)

    df_rate[col_name] = df_rate[col_name].fillna(0)

    return df_rate


def filter_connectivity(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    base = _coerce_and_filter(df, campaign_ids, months, branches)

    if base.empty:
        return pd.DataFrame(columns=["month", "day", "leads", "connect", "leads_cum", "conn_cum", "conn_per"])

    leads_col = _first_existing_col(base, ["leads", "total_leads"])
    conn_col = _first_existing_col(base, ["connect", "connect_value"])

    if leads_col is None or conn_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "leads", "connect", "leads_cum", "conn_cum", "conn_per"])

    base[leads_col] = pd.to_numeric(base[leads_col], errors="coerce").fillna(0)
    base[conn_col] = pd.to_numeric(base[conn_col], errors="coerce").fillna(0)

    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(
                leads=(leads_col, 'sum'),
                connect=(conn_col, 'sum')
            )
            .sort_values(['month', 'day'])
            .reset_index(drop=True)
    )

    agg["leads"] = pd.to_numeric(agg["leads"], errors="coerce").fillna(0)
    agg["connect"] = pd.to_numeric(agg["connect"], errors="coerce").fillna(0)

    agg["leads_cum"] = agg.groupby("month", group_keys=False)["leads"].cumsum()
    agg["conn_cum"] = agg.groupby("month", group_keys=False)["connect"].cumsum()

    # Calculate connectivity %
    agg["conn_per"] = pd.Series(np.where(
        agg["conn_cum"] > agg["leads_cum"],
        0,
        (agg["conn_cum"] / agg["leads_cum"].replace(0, pd.NA)) * 100
    ), index=agg.index).round(2).fillna(0)

    return agg


def filter_conversion_data1(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Expected input columns:
      month, campaign_id, branch, day, deal_closed, rv_3_year

    Output columns:
      month, day, deal_closed, rv_3_year
    """

    # ---------- Base filtering ----------
    base = _coerce_and_filter(df, campaign_ids, months, branches)


    if base.empty:
        return pd.DataFrame(
            columns=["month", "day", "deal_closed", "rv_3_year"]
        )

    # ---------- Type safety ----------
    base["deal_closed"] = pd.to_numeric(base["deal_closed"], errors="coerce").fillna(0)
    base["rv_3_year"] = pd.to_numeric(base["rv_3_year"], errors="coerce").fillna(0)

    # ---------- Aggregation ----------
    agg = (
        base
        .groupby(["month", "day"], as_index=False)
        .agg(
            deal_closed=("deal_closed", "sum"),
            rv_3_year=("rv_3_year", "sum")
        )
        .sort_values(["month", "day"])
    )

    return agg


def filter_call_scheduling_data(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Input columns:
      campaign_id, month, day, call_scheduling_count, leads, branch?
    Output columns:
      month, day, call_scheduling_cum, leads_cum
    """
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=["month", "day", "call_scheduling_cum", "leads_cum"])

    cs_col = _first_existing_col(base, ["call_scheduling_count"])
    leads_col = _first_existing_col(base, ["leads"])
    if cs_col is None or leads_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=["month", "day", "call_scheduling_cum", "leads_cum"])

    base[cs_col] = _to_float_series(base[cs_col]).fillna(0)
    base[leads_col] = _to_float_series(base[leads_col]).fillna(0)
    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(call_scheduling_cum=(cs_col, "sum"),
                 leads_cum=(leads_col, "sum"))
            .sort_values(["month", "day"])
    )
    return agg


def filter_call_scheduling_rate(df_cs, col_name):
    """
    Call Scheduling Rate = (call_scheduling_cum / leads_cum) * 100
    """
    if not {"month", "day", "call_scheduling_cum", "leads_cum"}.issubset(df_cs.columns):
        return pd.DataFrame(columns=["month", "day", col_name])

    df_cs_rate = df_cs.copy()

    df_cs_rate["call_scheduling_cum"] = pd.to_numeric(df_cs_rate["call_scheduling_cum"], errors="coerce").fillna(0)
    df_cs_rate["leads_cum"] = pd.to_numeric(df_cs_rate["leads_cum"], errors="coerce").fillna(0)

    den = df_cs_rate["leads_cum"].replace(0, np.nan)

    df_cs_rate[col_name] = pd.Series(np.where(
        df_cs_rate["call_scheduling_cum"] > den,
        0,
        (df_cs_rate["call_scheduling_cum"] / den) * 100
    ), index=df_cs_rate.index).round(2).fillna(0)

    df_cs_rate[col_name] = df_cs_rate[col_name].fillna(0)

    return df_cs_rate


def filter_tme_experience(
    df: pd.DataFrame,
    campaign_ids: List[int],
    months: List[int],
    branches: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Raw table columns: campaign_call_addor, call_addor, empname, department,
                       branch, month, day, empcode, experience

    Logic:
      1. Filter by campaign/month/branch.
      2. Drop duplicate TMEs per (month, day, empcode) so a TME is not
         counted twice in the same day across campaigns.
      3. Group by (month, day):
         - total_experience = sum of experience days
         - unique_tme       = nunique of empcode
      4. avg_experience     = total_experience / unique_tme
    """
    empty_cols = ["month", "day", "total_experience", "unique_tme", "avg_experience"]
    base = _coerce_and_filter(df, campaign_ids, months, branches)
    if base.empty:
        return pd.DataFrame(columns=empty_cols)

    emp_col = _first_existing_col(base, ["empcode", "emp_code"])
    exp_col = _first_existing_col(base, ["experience", "experience_days"])
    if emp_col is None or exp_col is None or not {"month", "day"}.issubset(base.columns):
        return pd.DataFrame(columns=empty_cols)

    base[exp_col] = _to_float_series(base[exp_col]).fillna(0)

    # Drop duplicate TMEs per (month, day, empcode)
    base = base.drop_duplicates(subset=["month", "day", emp_col])

    agg = (
        base.groupby(["month", "day"], as_index=False)
            .agg(
                total_experience=(exp_col, "sum"),
                unique_tme=(emp_col, "nunique"),
            )
            .sort_values(["month", "day"])
    )

    # Calculate average experience = total_experience / unique_tme
    den = agg["unique_tme"].replace(0, np.nan)
    agg["avg_experience"] = (
        agg["total_experience"]
        .div(den)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .round(2)
    )

    return agg