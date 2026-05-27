from flask import render_template, request, make_response
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from app.data_fetcher import fetch_table
from app.connection import create_conn_240
from app.filters import filter_rv_data, filter_rv_rate , filter_rv_data_closed


# --------------- Cache ---------------
_WA_CACHE = {"df": None, "fetched_at": None}
_FB_CACHE = {"df": None, "fetched_at": None}
_PMAX_CACHE = {"df": None, "fetched_at": None}
_MAILER_CACHE = {"df": None, "fetched_at": None}

# FB campaign IDs
FB_CAMPAIGN_IDS = [108, 159, 196, 285, 286]
FB_CAMPAIGN_NAMES = [
    "FB Digital Marketing Ad",
    "FB Digital Marketing FL",
    "FB Double Intent",
    "FB Advertise Warm",
    "FB Advertise Hot",
]

# P-Max campaign IDs
PMAX_CAMPAIGN_IDS = [111, 112]

# Mailer campaign IDs
MAILER_CAMPAIGN_IDS = [208 , 242 , 288]


def _fetch_fb_data(force_update=False):
    """Fetch FB cost data from dashboard_internal_meta_com_daywise."""
    global _FB_CACHE
    if force_update or _FB_CACHE["df"] is None:
        conn = create_conn_240()
        try:
            df = fetch_table("dashboard_internal_meta_com_daywise", conn)
        finally:
            conn.close()
        if 'log_date' in df.columns:
            df['log_date'] = pd.to_datetime(df['log_date'], errors='coerce')
            df['month'] = df['log_date'].dt.month
            df['day'] = df['log_date'].dt.day
        for col in ['amount', 'hot_leads']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'campaign_name' in df.columns:
            df['campaign_name'] = df['campaign_name'].astype(str).str.strip()
        _FB_CACHE["df"] = df
        _FB_CACHE["fetched_at"] = datetime.now()
    return _FB_CACHE["df"]


def _fetch_pmax_data(force_update=False):
    """Fetch P-Max cost data from dashboard_internal_pmax_com_daywise."""
    global _PMAX_CACHE
    if force_update or _PMAX_CACHE["df"] is None:
        conn = create_conn_240()
        try:
            df = fetch_table("dashboard_internal_pmax_com_daywise", conn)
        finally:
            conn.close()
        if 'log_date' in df.columns:
            df['log_date'] = pd.to_datetime(df['log_date'], errors='coerce')
            df['month'] = df['log_date'].dt.month
            df['day'] = df['log_date'].dt.day
        for col in ['amount', 'hot_leads']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'campaign_name' in df.columns:
            df['campaign_name'] = df['campaign_name'].astype(str).str.strip()
        _PMAX_CACHE["df"] = df
        _PMAX_CACHE["fetched_at"] = datetime.now()
    return _PMAX_CACHE["df"]


def _fetch_wa_data(force_update=False):
    global _WA_CACHE
    if force_update or _WA_CACHE["df"] is None:
        conn = create_conn_240()
        try:
            df = fetch_table("dashboard_internal_wa_com", conn)
        finally:
            conn.close()
        # Standardise columns
        if 'log_date' in df.columns:
            df['log_date'] = pd.to_datetime(df['log_date'], errors='coerce')
            df['month'] = df['log_date'].dt.month
            df['day'] = df['log_date'].dt.day
        for col in ['leads_count', 'delivered_count', 'cost']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'cost/leads' in df.columns:
            df['cost_per_lead'] = pd.to_numeric(df['cost/leads'], errors='coerce')
        elif 'cost' in df.columns and 'leads_count' in df.columns:
            df['cost_per_lead'] = df['cost'] / df['leads_count'].replace(0, pd.NA)
        if 'branch' in df.columns:
            df['branch'] = df['branch'].astype(str).str.strip()
        if 'campaign_name' in df.columns:
            df['campaign_name'] = df['campaign_name'].astype(str).str.strip()
            df['campaign_name'] = df['campaign_name'].replace('customoffer', 'marketing pilot A')
        _WA_CACHE["df"] = df
        _WA_CACHE["fetched_at"] = datetime.now()
    return _WA_CACHE["df"]


def _filter_wa_data(df, campaigns, months, branches):
    """Filter WhatsApp cost data by campaign_name, month, branch."""
    filtered = df.copy()
    if campaigns:
        filtered = filtered[filtered['campaign_name'].isin(campaigns)]
    if months:
        filtered = filtered[filtered['month'].isin(months)]
    if branches:
        bl = [b.strip().lower() for b in branches]
        filtered = filtered[filtered['branch'].str.lower().isin(bl)]
    return filtered

def _fetch_mailer_data(force_update=False):
    global _MAILER_CACHE
    if force_update or _MAILER_CACHE["df"] is None:
        conn = create_conn_240()
        try:
            df = fetch_table("dashboard_internal_mailer_com", conn)
        finally:
            conn.close()
        # Standardise columns
        if 'log_date' in df.columns:
            df['log_date'] = pd.to_datetime(df['log_date'], errors='coerce')
            df['month'] = df['log_date'].dt.month
            df['day'] = df['log_date'].dt.day
        for col in ['leads_count', 'sent_count', 'cost']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'cost/leads' in df.columns:
            df['cost_per_lead'] = pd.to_numeric(df['cost/leads'], errors='coerce')
        elif 'cost' in df.columns and 'leads_count' in df.columns:
            df['cost_per_lead'] = df['cost'] / df['leads_count'].replace(0, pd.NA)
        if 'branch' in df.columns:
            df['branch'] = df['branch'].astype(str).str.strip()
        if 'campaign_name' in df.columns:
            df['campaign_name'] = df['campaign_name'].astype(str).str.strip()
            df['campaign_name'] = df['campaign_name'].replace('customoffer', 'marketing pilot A')
        _MAILER_CACHE["df"] = df
        _MAILER_CACHE["fetched_at"] = datetime.now()
    return _MAILER_CACHE["df"]


def _filter_mailer_data(df, campaigns, months, branches):
    """Filter Mailer cost data by campaign_name, month, branch."""
    filtered = df.copy()
    if campaigns:
        filtered = filtered[filtered['campaign_name'].isin(campaigns)]
    if months:
        filtered = filtered[filtered['month'].isin(months)]
    if branches:
        bl = [b.strip().lower() for b in branches]
        filtered = filtered[filtered['branch'].str.lower().isin(bl)]
    return filtered


def _make_cumulative_line_chart(df, ycol, title, ytitle, months):
    """Cumulative day-wise line chart: cumsum of ycol per month."""
    fig = go.Figure()
    if df is None or df.empty or ycol not in df.columns:
        fig.update_layout(
            annotations=[dict(text=f"No data for '{ycol}'.", x=0.5, y=0.5, showarrow=False)],
            height=420
        )
        return fig

    MONTH_NAME = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    color_palette = ["#4983f6", "#38cb38", "#fd3a3a", "#e2a60e", "#e377c2", "#bcbd22", "#8c564b"]

    if 'month' in df.columns and months:
        for idx, m in enumerate(months):
            mdf = df[df['month'] == m].sort_values('day')
            if not mdf.empty:
                day_agg = mdf.groupby('day')[ycol].sum().reset_index().sort_values('day')
                day_agg['cumulative'] = day_agg[ycol].cumsum()
                color = color_palette[idx % len(color_palette)]
                fig.add_trace(go.Scatter(
                    x=day_agg['day'], y=day_agg['cumulative'],
                    mode='lines+markers',
                    name=MONTH_NAME.get(m, f"Month {m}"),
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=6)
                ))

    fig.update_layout(
        xaxis_title="Day of Month",
        yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        showlegend=True,
        margin=dict(l=50, r=30, t=60, b=90),
        height=420
    )
    return fig


def _make_cumulative_cpl_chart(df, months, cost_of_sale=0):
    """Cumulative Cost Per Lead = cumulative cost / cumulative leads, day-wise."""
    fig = go.Figure()
    if df is None or df.empty or 'cost' not in df.columns or 'leads_count' not in df.columns:
        fig.update_layout(
            annotations=[dict(text="No data for cost/lead.", x=0.5, y=0.5, showarrow=False)],
            height=420
        )
        return fig

    MONTH_NAME = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    color_palette = ["#4983f6", "#38cb38", "#fd3a3a", "#e2a60e", "#e377c2", "#bcbd22", "#8c564b"]

    if 'month' in df.columns and months:
        for idx, m in enumerate(months):
            mdf = df[df['month'] == m].sort_values('day')
            if not mdf.empty:
                day_agg = mdf.groupby('day')[['cost', 'leads_count']].sum().reset_index().sort_values('day')
                day_agg['cum_cost'] = day_agg['cost'].cumsum()
                day_agg['cum_leads'] = day_agg['leads_count'].cumsum()
                day_agg['cum_cpl'] = day_agg['cum_cost'] / day_agg['cum_leads'].replace(0, pd.NA)
                if cost_of_sale:
                    day_agg['cum_cpl'] = day_agg['cum_cpl'] + cost_of_sale
                color = color_palette[idx % len(color_palette)]
                fig.add_trace(go.Scatter(
                    x=day_agg['day'], y=day_agg['cum_cpl'],
                    mode='lines+markers',
                    name=MONTH_NAME.get(m, f"Month {m}"),
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=6)
                ))

    cpl_label = "Cost / Lead" if not cost_of_sale else f"Cost / Lead (+ {cost_of_sale:g} Cost of Sale)"
    fig.update_layout(
        xaxis_title="Day of Month",
        yaxis_title=cpl_label,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        showlegend=True,
        margin=dict(l=50, r=30, t=60, b=90),
        height=420
    )
    return fig


def _make_cpl_rv_combined_chart(wa_df, rv_df, months, cost_of_sale=0):
    """
    Combined single-axis line chart:
      - Solid line: Cumulative Cost Per Lead
      - Dashed line: RV Per Lead
    Both on the same y-axis scale, per month.
    """
    fig = go.Figure()

    has_cpl = wa_df is not None and not wa_df.empty and {'cost', 'leads_count', 'month', 'day'}.issubset(wa_df.columns)
    has_rv = rv_df is not None and not rv_df.empty and {'RV_Lead_Rate', 'month', 'day'}.issubset(rv_df.columns)

    if not has_cpl and not has_rv:
        fig.update_layout(
            annotations=[dict(text="No data available.", x=0.5, y=0.5, showarrow=False)],
            height=450
        )
        return fig

    MONTH_NAME = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    color_palette = ["#4983f6", "#38cb38", "#fd3a3a", "#e2a60e", "#e377c2", "#bcbd22", "#8c564b"]

    for idx, m in enumerate(months):
        color = color_palette[idx % len(color_palette)]
        month_label = MONTH_NAME.get(m, f"Month {m}")

        # Find the max day for RV per lead data in this month
        rv_max_day = 0
        if has_rv:
            rdf = rv_df[rv_df['month'] == m].sort_values('day')
            if not rdf.empty:
                rv_max_day = int(rdf['day'].max())

        # Solid line: Cumulative Cost Per Lead (extended to match RV per lead range)
        if has_cpl:
            mdf = wa_df[wa_df['month'] == m].sort_values('day')
            if not mdf.empty:
                day_agg = mdf.groupby('day')[['cost', 'leads_count']].sum().reset_index().sort_values('day')
                day_agg['cum_cost'] = day_agg['cost'].cumsum()
                day_agg['cum_leads'] = day_agg['leads_count'].cumsum()
                day_agg['cum_cpl'] = day_agg['cum_cost'] / day_agg['cum_leads'].replace(0, pd.NA)
                if cost_of_sale:
                    day_agg['cum_cpl'] = day_agg['cum_cpl'] + cost_of_sale

                # Extend CPL line to match RV per lead's last day
                last_cpl_day = int(day_agg['day'].max())
                if rv_max_day > last_cpl_day:
                    last_cpl_value = day_agg['cum_cpl'].iloc[-1]
                    extend_days = list(range(last_cpl_day + 1, rv_max_day + 1))
                    extend_df = pd.DataFrame({'day': extend_days, 'cum_cpl': last_cpl_value})
                    day_agg = pd.concat([day_agg[['day', 'cum_cpl']], extend_df], ignore_index=True)

                fig.add_trace(go.Scatter(
                    x=day_agg['day'], y=day_agg['cum_cpl'],
                    mode='lines+markers',
                    name=f"{month_label} — Cost/Lead",
                    line=dict(color=color, width=2, dash='solid'),
                    marker=dict(color=color, size=6, symbol='circle'),
                    legendgroup=month_label
                ))

        # Dashed line: RV Per Lead
        if has_rv and not rdf.empty:
            fig.add_trace(go.Scatter(
                x=rdf['day'], y=rdf['RV_Lead_Rate'],
                mode='lines+markers',
                name=f"{month_label} — RV/Lead",
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(color=color, size=7, symbol='diamond'),
                legendgroup=month_label
            ))

    fig.update_layout(
        xaxis_title="Day of Month",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        showlegend=True,
        margin=dict(l=50, r=30, t=60, b=90),
        height=450
    )
    return fig


def _make_meta_cpl_rv_combined_chart(cost_df, rv_df, months, label="FB", cost_of_sale=0):
    """
    Combined single-axis line chart:
      - Solid line: Cumulative Cost Per Lead (amount / hot_leads + cost_of_sale)
      - Dashed line: RV Per Lead
    Both on the same y-axis scale, per month.
    """
    fig = go.Figure()

    has_cpl = cost_df is not None and not cost_df.empty and {'amount', 'hot_leads', 'month', 'day'}.issubset(cost_df.columns)
    has_rv = rv_df is not None and not rv_df.empty and {'RV_Lead_Rate', 'month', 'day'}.issubset(rv_df.columns)

    if not has_cpl and not has_rv:
        fig.update_layout(
            annotations=[dict(text=f"No {label} data available.", x=0.5, y=0.5, showarrow=False)],
            height=450
        )
        return fig

    MONTH_NAME = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    color_palette = ["#4983f6", "#38cb38", "#fd3a3a", "#e2a60e", "#e377c2", "#bcbd22", "#8c564b"]

    for idx, m in enumerate(months):
        color = color_palette[idx % len(color_palette)]
        month_label = MONTH_NAME.get(m, f"Month {m}")

        # Find the max day for RV per lead data in this month
        rv_max_day = 0
        rdf = pd.DataFrame()
        if has_rv:
            rdf = rv_df[rv_df['month'] == m].sort_values('day')
            if not rdf.empty:
                rv_max_day = int(rdf['day'].max())

        # Solid line: Cumulative Cost Per Lead (extended to match RV range)
        if has_cpl:
            mdf = cost_df[cost_df['month'] == m].sort_values('day')
            if not mdf.empty:
                day_agg = mdf.groupby('day')[['amount', 'hot_leads']].sum().reset_index().sort_values('day')
                day_agg['cum_amount'] = day_agg['amount'].cumsum()
                day_agg['cum_leads'] = day_agg['hot_leads'].cumsum()
                day_agg['cum_cpl'] = day_agg['cum_amount'] / day_agg['cum_leads'].replace(0, pd.NA)
                if cost_of_sale:
                    day_agg['cum_cpl'] = day_agg['cum_cpl'] + cost_of_sale

                # Extend CPL line to match RV per lead's last day
                last_cpl_day = int(day_agg['day'].max())
                if rv_max_day > last_cpl_day:
                    last_cpl_value = day_agg['cum_cpl'].iloc[-1]
                    extend_days = list(range(last_cpl_day + 1, rv_max_day + 1))
                    extend_df = pd.DataFrame({'day': extend_days, 'cum_cpl': last_cpl_value})
                    day_agg = pd.concat([day_agg[['day', 'cum_cpl']], extend_df], ignore_index=True)

                fig.add_trace(go.Scatter(
                    x=day_agg['day'], y=day_agg['cum_cpl'],
                    mode='lines+markers',
                    name=f"{month_label} — Cost/Lead",
                    line=dict(color=color, width=2, dash='solid'),
                    marker=dict(color=color, size=6, symbol='circle'),
                    legendgroup=month_label
                ))

        # Dashed line: RV Per Lead
        if has_rv and not rdf.empty:
            fig.add_trace(go.Scatter(
                x=rdf['day'], y=rdf['RV_Lead_Rate'],
                mode='lines+markers',
                name=f"{month_label} — RV/Lead",
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(color=color, size=7, symbol='diamond'),
                legendgroup=month_label
            ))

    fig.update_layout(
        xaxis_title="Day of Month",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        showlegend=True,
        margin=dict(l=50, r=30, t=60, b=90),
        height=450
    )
    return fig

def _make_combined_com_curve(
    wa_df,
    fb_df,
    pmax_df,
    mailer_df,
    rv_df,
    months
):
    """
    Combined COM Curve

    COM = Cost / RV

    WA + FB + PMAX + MAILER combined
    """

    fig = go.Figure()

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
        "#bcbd22"
    ]

    for idx, m in enumerate(months):

        color = color_palette[idx % len(color_palette)]
        month_label = MONTH_NAME.get(m, f"Month {m}")

        # ---------------- WA ----------------
        wa_mdf = wa_df[wa_df['month'] == m]

        wa_day = pd.DataFrame()

        if not wa_mdf.empty:

            wa_day = (
                wa_mdf.groupby('day')[['cost']]
                .sum()
                .reset_index()
            )

            wa_day.rename(columns={
                'cost': 'amount'
            }, inplace=True)

        # ---------------- FB ----------------
        fb_mdf = fb_df[fb_df['month'] == m]

        fb_day = pd.DataFrame()

        if not fb_mdf.empty:

            fb_day = (
                fb_mdf.groupby('day')[['amount']]
                .sum()
                .reset_index()
            )

        # ---------------- PMAX ----------------
        pmax_mdf = pmax_df[pmax_df['month'] == m]

        pmax_day = pd.DataFrame()

        if not pmax_mdf.empty:

            pmax_day = (
                pmax_mdf.groupby('day')[['amount']]
                .sum()
                .reset_index()
            )

        # ---------------- MAILER ----------------
        mailer_mdf = mailer_df[mailer_df['month'] == m]

        mailer_day = pd.DataFrame()

        if not mailer_mdf.empty:

            mailer_day = (
                mailer_mdf.groupby('day')[['cost']]
                .sum()
                .reset_index()
            )

            mailer_day.rename(columns={
                'cost': 'amount'
            }, inplace=True)

        # ---------------- COMBINED COST ----------------
        combined_cost = pd.concat(
            [
                wa_day,
                fb_day,
                pmax_day,
                mailer_day
            ],
            ignore_index=True
        )

        if combined_cost.empty:
            continue

        combined_cost = (
            combined_cost.groupby('day')[['amount']]
            .sum()
            .reset_index()
            .sort_values('day')
        )

        # ---------------- RV ----------------
        rdf = rv_df[rv_df['month'] == m]

        if rdf.empty:
            continue

        rv_day = (
            rdf.groupby('day')[["rv_3_yr_hot_team_cum"]]
            .sum()
            .reset_index()
            .sort_values('day')
        )

        # ---------------- MERGE ----------------
        final_df = pd.merge(
            combined_cost,
            rv_day,
            on='day',
            how='outer'
        ).fillna(0)

        final_df = final_df.sort_values('day')

        # ---------------- CUMULATIVE ----------------
        final_df['cum_cost'] = final_df['amount'].cumsum()

        final_df["rv_3_yr_hot_team_cum"] = pd.to_numeric(
            final_df["rv_3_yr_hot_team_cum"],
            errors="coerce"
        ).fillna(0)

        final_df['cum_rv'] = final_df[
            'rv_3_yr_hot_team_cum'
        ]

        # ---------------- COM ----------------
        final_df['com'] = (
            final_df['cum_cost']
            / final_df['cum_rv'].replace(0, pd.NA)
        )*100

        final_df = final_df[final_df['com'] > 0]

        # Uncomment below line if percentage needed
        # final_df['com'] = final_df['com'] * 100

        # ---------------- STATIC CURVE ----------------
        max_day = int(final_df['day'].max())

        all_days = pd.DataFrame({
            'day': range(1, max_day + 1)
        })

        final_df = pd.merge(
            all_days,
            final_df[['day', 'com']],
            on='day',
            how='left'
        )

        final_df['com'] = final_df['com'].ffill()

        # ---------------- PLOT ----------------
        fig.add_trace(go.Scatter(
            x=final_df['day'],
            y=final_df['com'],
            mode='lines+markers',
            name=f"{month_label} COM",
            line=dict(
                color=color,
                width=3
            ),
            marker=dict(
                color=color,
                size=6
            )
        ))

    fig.update_layout(
        title="Combined COM Curve",
        xaxis_title="Day of Month",
        yaxis_title="Cost / RV",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(
            l=50,
            r=30,
            t=60,
            b=90
        ),
        height=450
    )

    return fig


def cost_analysis():
    """Route handler for cost analysis page."""

    is_update_click = request.method == 'POST' and request.form.get('update_data') == '1'
    is_filters_click = request.method == 'POST' and not is_update_click

    df = _fetch_wa_data(force_update=is_update_click)
    last_updated = _WA_CACHE["fetched_at"].strftime("%d-%b-%Y %I:%M %p") if _WA_CACHE["fetched_at"] else "Never"

    # Build choices from data
    campaign_choices = sorted(df['campaign_name'].dropna().unique().tolist()) if 'campaign_name' in df.columns else []
    branch_choices = sorted(df['branch'].dropna().unique().tolist()) if 'branch' in df.columns else []

    MONTH_NAME_MAP = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    MONTH_NAME_REVERSE = {v: k for k, v in MONTH_NAME_MAP.items()}

    # Show last 4 months dynamically
    current_month = datetime.now().month
    allowed_months = [(current_month - i - 1) % 12 + 1 for i in range(3, -1, -1)]
    month_choices = [MONTH_NAME_MAP[m] for m in allowed_months]

    # Read filters
    if is_filters_click:
        campaigns_selected = request.form.getlist('campaign_name')
        month_names_selected = request.form.getlist('month')
        branches_selected = request.form.getlist('branch')
        try:
            cost_of_sale = float(request.form.get('cost_of_sale', 0) or 0)
        except (ValueError, TypeError):
            cost_of_sale = 0.0
    else:
        campaigns_selected = []
        month_names_selected = []
        branches_selected = []
        cost_of_sale = 0.0

    months_selected = [MONTH_NAME_REVERSE[n] for n in month_names_selected if n in MONTH_NAME_REVERSE]

    # Default to all allowed months when none selected so charts render on first load
    if not months_selected:
        months_selected = allowed_months[:]
        month_names_selected = [MONTH_NAME_MAP[m] for m in months_selected]

    # Display strings
    selected_campaigns_display = ", ".join(campaigns_selected) if campaigns_selected else "None"
    selected_months_display = ", ".join(month_names_selected) if month_names_selected else "None"
    selected_branches_display = ", ".join(branches_selected) if branches_selected else "None"

    # Filter WA data
    filtered = _filter_wa_data(df, campaigns_selected, months_selected, branches_selected)

    # --- RV Per Lead from conversion table ---
    # Campaign ID to group mapping for RV per lead
    RV_CAMPAIGN_IDS = [200, 160, 256, 258, 243 , 244 , 233]
    combined_rv_campaigns = (
    RV_CAMPAIGN_IDS +
    FB_CAMPAIGN_IDS +
    PMAX_CAMPAIGN_IDS +
    MAILER_CAMPAIGN_IDS
    )

    from app.routes import _get_data, _DATA_CACHE, _standardize_columns, _prepare_df
    dfs = _get_data(force_update=is_update_click)
    conversion_data = dfs['conversion']
    rv_deal_closed = dfs['conversion_deal']
    # conversion_data.rename(columns={'rv_3_year':'rv_3_yr_hot_team'}, inplace=True)
    filtered_rv_lead = filter_rv_data(
        conversion_data, RV_CAMPAIGN_IDS, months_selected, branches_selected
    )
    filtered_rv_lead_rate = filter_rv_rate(filtered_rv_lead, 'RV_Lead_Rate')
    filtered_rv_lead_rate = _prepare_df(filtered_rv_lead_rate)

    combined_rv = filter_rv_data_closed(
    rv_deal_closed,
    combined_rv_campaigns,
    months_selected,
    branches=None
    )
    combined_rv = _prepare_df(combined_rv)

    # Charts (WA)
    fig_cost_cum = _make_cumulative_line_chart(filtered, 'cost', "Cost of Marketing", "Cost of Marketing", months_selected)
    fig_cpl_cum = _make_cumulative_cpl_chart(filtered, months_selected, cost_of_sale)
    fig_cpl_rv = _make_cpl_rv_combined_chart(filtered, filtered_rv_lead_rate, months_selected, cost_of_sale)
    # --- FB Cost/Lead and RV/Lead (Pan India, no branch filter) ---
    fb_df = _fetch_fb_data(force_update=is_update_click)
    # Use all rows from the Meta table (already FB-specific), only filter by month
    fb_filtered = fb_df.copy()
    if months_selected:
        fb_filtered = fb_filtered[fb_filtered['month'].isin(months_selected)]

    # FB RV from conversion table using FB campaign IDs (Pan India = no branch filter)
    fb_rv_lead = filter_rv_data(conversion_data, FB_CAMPAIGN_IDS, months_selected, branches=None)
    fb_rv_lead_rate = filter_rv_rate(fb_rv_lead, 'RV_Lead_Rate')
    fb_rv_lead_rate = _prepare_df(fb_rv_lead_rate)

    fig_fb_cpl_rv = _make_meta_cpl_rv_combined_chart(fb_filtered, fb_rv_lead_rate, months_selected, label="FB", cost_of_sale=cost_of_sale)

    # --- P-Max Cost/Lead and RV/Lead (Pan India, no branch filter) ---
    pmax_df = _fetch_pmax_data(force_update=is_update_click)
    pmax_filtered = pmax_df.copy()
    if months_selected:
        pmax_filtered = pmax_filtered[pmax_filtered['month'].isin(months_selected)]
    
    # --- MAILER Cost Data ---
    mailer_df = _fetch_mailer_data(force_update=is_update_click)

    mailer_filtered = mailer_df.copy()

    if months_selected:
        mailer_filtered = mailer_filtered[
            mailer_filtered['month'].isin(months_selected)
        ]

    pmax_rv_lead = filter_rv_data(conversion_data, PMAX_CAMPAIGN_IDS, months_selected, branches=None)
    pmax_rv_lead_rate = filter_rv_rate(pmax_rv_lead, 'RV_Lead_Rate')
    pmax_rv_lead_rate = _prepare_df(pmax_rv_lead_rate)

    fig_combined_com = _make_combined_com_curve(
    filtered,
    fb_filtered,
    pmax_filtered,
    mailer_filtered,
    combined_rv,
    months_selected
    )

    fig_pmax_cpl_rv = _make_meta_cpl_rv_combined_chart(pmax_filtered, pmax_rv_lead_rate, months_selected, label="P-Max", cost_of_sale=cost_of_sale)

    html = render_template(
        'cost_analysis.html',
        page="cost_analysis",
        refreshed=is_update_click,
        last_updated=last_updated,

        campaign_choices=campaign_choices,
        campaigns_selected=campaigns_selected,

        month_choices=month_choices,
        month_selected=month_names_selected,

        branch_choices=branch_choices,
        branches_selected=branches_selected,

        selected_campaigns_display=selected_campaigns_display,
        selected_months_display=selected_months_display,
        selected_branches_display=selected_branches_display,
        cost_of_sale=cost_of_sale,

        graph_cost_cum=fig_cost_cum.to_html(full_html=False, include_plotlyjs='cdn'),
        graph_cpl_cum=fig_cpl_cum.to_html(full_html=False, include_plotlyjs=False),
        graph_cpl_rv=fig_cpl_rv.to_html(full_html=False, include_plotlyjs=False),
        graph_fb_cpl_rv=fig_fb_cpl_rv.to_html(full_html=False, include_plotlyjs=False),
        graph_pmax_cpl_rv=fig_pmax_cpl_rv.to_html(full_html=False, include_plotlyjs=False),
        graph_combined_com=fig_combined_com.to_html(
        full_html=False,
        include_plotlyjs=False )
    )


    resp = make_response(html)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp
