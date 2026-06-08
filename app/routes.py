from flask import render_template, request, make_response
import pandas as pd
import numpy as np
from datetime import datetime
from app import application
from app.data_fetcher import fetch_tables_as_dict
from app.branch_utilis import filter_data, plot_grouped_bar, filter_data_conv, pan_india , filter_appointment_rate_branch , pan_india_rate
from app.filters import (
    filter_conversion_data,
    filter_lead_generation_data,
    filter_appointment_data,
    filter_appointment_rate,
    filter_NIB_data,
    filter_nib_rate,
    filter_attempt_data,
    filter_ni_rate,
    filter_ni_data,
    filter_connectivity,
    filter_attempt_rate,
    filter_conversion_data1,
    filter_tme_experience,
    filter_bad_dispo_data,
    filter_bad_dispo_rate,
    filter_call_scheduling_data,
    filter_call_scheduling_rate,
    filter_rv_data,
    filter_rv_rate
)
from app.plots import make_monthly_line_bar, make_timeseries_figure
from app.campaign import campaign_analysis
from app.cost_analysis import cost_analysis

# ---------------- Cache helpers (unchanged) ----------------
_DATA_CACHE = {"dfs": None, "fetched_at": None}
import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
dict_camp = {
    "Whatsapp Performance Marketing": 160,
    "WA Double Intent": 200,
    "FB Digital Marketing Ad": 108,
    "FB Digital Marketing FL": 159,
    "FB Double Intent": 196,
    "P-Max Hot": 111,
    "P-Max Warm": 112,
    "Competitor FOMO Click | WA": 258,
    "Competitor FOMO Super | WA": 259,
    "Category Surge Super | Email": 208,
    "Category Surge Click | Email": 173,
    "Category Surge Open | Email": 172,
    "Seasonal Campaign SUPER | Email": 242,
    "Seasonal Campaign Click | Email": 104,
    "Seasonal Campaign Open | Email": 103,
    "Marketing Pilot A | WA": 243,
    "Marketing Pilot B | WA": 244,
    "New Seasonal Spike | WA": 256,
    "Customer Intent | WA": 257,
    "Marketing Email | Super": 288,
    "FB Advertise Warm": 285,
    "FB Advertise Hot": 286,
    "Offer Notification": 260,
    "Introductory Price | WA": 233,
    "G Double Intent Hot": 198,
    "G Double Intent Warm": 199,
    "Double Intent Offer | WA": 197,
    "Testimonial Click | Email": 240,
    "Performance Marketing JDA | WA": 261,
    "Marketing Ad Now Hot | Email": 117,
    "MSME Educational Click | Email": 191,
    "MSME Myth Busters Open | Email": 193,
    'Paid_Expired': 189,
    'LBC' : 319,
    'user feedback click' : 22,
    'User feedback super hot' : 178
}

dict_camp_product = {
    "JD Biz": 80,
    "Jd Biz Notification": 190,
    "JD Biz Super Hot": 149,
    "JD Biz Claimed Click | Email": 150,
    "Edit Listing Owner": 3,
    "Leads Section": 77,
    "New Free Listing Hot": 97,
    "Free Leads Super Hot | WA": 184,
    "Free Leads Click | WA": 187,
    "Free Leads High Intent": 282,
    "New Advertise Hot": 101,
    "New Advertise Warm": 102,
    "User Feedback Super | Email": 178,
    "User Feedback Click | Email": 22
}


# Croos verification of campaign name to ID mapping and vice versa

def _campaign_name_to_ids(selected_names, camp_dict):
    return [camp_dict[name] for name in selected_names if name in camp_dict]

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if 'month_no' in df.columns and 'month' not in df.columns:
        df.rename(columns={'month_no': 'month'}, inplace=True)
    if 'no_days' in df.columns and 'day' not in df.columns:
        df.rename(columns={'no_days': 'day'}, inplace=True)
    if 'category' not in df.columns:
        for c in ['Category Name', 'category_name', 'Category', 'Category_Name']:
            if c in df.columns:
                df.rename(columns={c: 'category'}, inplace=True)
                break
    if 'month' in df.columns:
        df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
    if 'day' in df.columns:
        df['day'] = pd.to_numeric(df['day'], errors='coerce').astype('Int64')
    return df


def _fetch_from_db():
    tables = [
        'dashboard_internal_conversion',
        'dashboard_internal_converted_deal_closed_date',
        'dashboard_internal_lead_generation',
        'dashboard_internal_appointment',
        'dashboard_internal_nib_data',
        'dashboard_internal_ni_data',
        'dashboard_internal_new_attempt_logic_data',
        'dashboard_internal_new_connectivity_logic_data',
        'dashboard_internal_tme_experience',
        'dashboard_internal_bad_disposition',
        'dashboard_internal_call_scheduling_data'
    ]
    TABLE_KEY_MAPPING = {
        "dashboard_internal_conversion": "conversion",
        "dashboard_internal_converted_deal_closed_date": "conversion_deal",
        "dashboard_internal_lead_generation": "lead_generation",
        "dashboard_internal_appointment": "appointment",
        "dashboard_internal_nib_data": "nib",
        "dashboard_internal_ni_data": "ni",
        'dashboard_internal_new_attempt_logic_data': "attempt",
        "dashboard_internal_new_connectivity_logic_data": "connectivity",
        "dashboard_internal_tme_experience": "tme_experience",
        "dashboard_internal_bad_disposition": "bad_dispo",
        "dashboard_internal_call_scheduling_data": "call_scheduling"
    }
    data = fetch_tables_as_dict(tables, table_key_mapping=TABLE_KEY_MAPPING)

    return {
        'appointment': _standardize_columns(data.get('appointment')),
        'lead': _standardize_columns(data.get('lead_generation')),
        'conversion': _standardize_columns(data.get('conversion')),
        'conversion_deal' : _standardize_columns(data.get('conversion_deal')),
        'NIB': _standardize_columns(data.get('nib')),
        'NI': _standardize_columns(data.get('ni')),
        'attempt': _standardize_columns(data.get('attempt')),
        'connectivity': _standardize_columns(data.get('connectivity')),
        'tme_experience': _standardize_columns(data.get('tme_experience')),
        'bad_dispo': _standardize_columns(data.get('bad_dispo')),
        'call_scheduling': _standardize_columns(data.get('call_scheduling'))
    }

def _get_data(force_update: bool = False):
    global _DATA_CACHE
    if force_update or _DATA_CACHE["dfs"] is None:
        dfs = _fetch_from_db()
        _DATA_CACHE["dfs"] = dfs
        _DATA_CACHE["fetched_at"] = datetime.now()
    return _DATA_CACHE["dfs"]

def _campaign_choices(camp_dict: dict):
    return sorted(camp_dict.keys())

def _choices_from_data(dfs: dict):
    """Build dropdown choices for campaign_id, month, branch from all tabs."""
    campaigns, months, branches = set(), set(), set()
    for df in dfs.values():
        if df is None or df.empty:
            continue
        if 'campaign_id' in df.columns:
            campaigns.update(pd.to_numeric(df['campaign_id'], errors='coerce').dropna().astype(int).tolist())
        if 'month' in df.columns:
            months.update(pd.to_numeric(df['month'], errors='coerce').dropna().astype(int).tolist())
        if 'branch' in df.columns:
            branches.update(df['branch'].dropna().astype(str).str.strip().tolist())

    campaign_list = sorted(campaigns)
    month_list    = sorted(months)
    branch_list   = sorted(branches)
    return campaign_list, month_list, branch_list


def _max_day_across(dfs: dict) -> int:
    max_day = 0
    for df in dfs.values():
        if df is None or df.empty or 'day' not in df.columns:
            continue
        d = pd.to_numeric(df['day'], errors='coerce').dropna()
        if not d.empty:
            max_day = max(max_day, int(d.max()))
    return max_day or 1


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dtypes for month/day if present, drop null month/day rows if both exist, and sort."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['month', 'day'])
    df = _standardize_columns(df)
    if {'month', 'day'}.issubset(df.columns):
        df = df.dropna(subset=['month', 'day']).sort_values(['month', 'day'])
    return df


# ---------------- Route ----------------
@application.route('/cost-analysis', methods=['GET', 'POST'])
def cost_analysis_page():
    return cost_analysis()

@application.route('/campaign', methods=['GET', 'POST'])
def branch_raw():
    return campaign_analysis()
@application.route('/branch', methods=['GET', 'POST'])
def branch_dashboard():
    """
    Two forms on this page:
      1) Top 'Update Data' form -> posts 'update_data=1'
      2) Filters form (campaign/day) -> posts 'campaign_id', 'day'
    """

    # Determine form action
    is_update_click = request.method == 'POST' and request.form.get('update_data') == '1'
    is_filters_click = request.method == 'POST' and not is_update_click

    # Fetch Data
    dfs = _get_data(force_update=is_update_click)
    last_updated = _DATA_CACHE["fetched_at"].strftime("%d-%b-%Y %I:%M %p") if _DATA_CACHE["fetched_at"] else "Never"

    appointment_df = dfs['appointment']
    lead_df        = dfs['lead']
    conversion_df  = dfs['conversion']
    conversion_df1 = dfs['conversion_deal']
    NIB_data       = dfs['NIB']
    NI_data        = dfs['NI']
    attempt        = dfs['attempt']
    conn           = dfs['connectivity']
    bad_dispo_data = dfs['bad_dispo']
    call_scheduling = dfs['call_scheduling']

    # ------------------------------------
    # CAMPAIGN DROPDOWN FIX (NAME → ID)
    # ------------------------------------
    campaign_name_choices = sorted(dict_camp.keys())          # show names in UI

    def campaign_names_to_ids(selected_names):
        """Convert selected campaign names from dropdown to IDs"""
        return [dict_camp[name] for name in selected_names if name in dict_camp]

    # Max day for default
    default_day = _max_day_across(dfs)

    # Read filter form
    if is_filters_click:
        selected_campaign_names = request.form.getlist('campaign_id')  # names
        selected_campaign_ids = campaign_names_to_ids(selected_campaign_names)
        day_raw = request.form.get('day')
    else:
        selected_campaign_names = []  # defaults
        selected_campaign_ids = []   # default to NONE – user must select
        day_raw = None

    # Day Selection
    try:
        selected_day = int(float(day_raw)) if day_raw else default_day
    except:
        selected_day = default_day

    # Defensive copies
    filtered_conv = conversion_df.copy()
    filtered_conv1 = conversion_df1.copy()
    filtered_lead = lead_df.copy()
    filtered_app  = appointment_df.copy()
    filtered_nib  = NIB_data.copy()
    filtered_ni   = NI_data.copy()
    filtered_attempt = attempt.copy()
    filtered_conn = conn.copy()
    filtered_bad_dispo = bad_dispo_data.copy()
    filtered_call_scheduling = call_scheduling.copy()

    # Ensure campaign_id dtype is numeric
    for df in (filtered_conv, filtered_lead, filtered_app, filtered_nib, filtered_ni, filtered_attempt, filtered_conn, filtered_bad_dispo, filtered_call_scheduling):
        if not df.empty and 'campaign_id' in df.columns:
            df['campaign_id'] = pd.to_numeric(df['campaign_id'], errors='coerce').astype('Int64')

    # ------------ APPLY FILTERS USING IDS ------------
    filtered_app.rename(columns={'appointment_close': 'appointment'}, inplace=True)
    grouped_conv = filter_data_conv(filtered_conv, day=selected_day, campaign=selected_campaign_ids)
    grouped_deal = filter_data(filtered_conv, day=selected_day, campaign=selected_campaign_ids,
                               target_col="hot_team_leads", flag=True)
    grouped_deal1 = filter_data(filtered_conv1, day=selected_day, campaign=selected_campaign_ids,
                               target_col="deal_closed", flag=True)
    grouped_rv =   filter_data(filtered_conv, day=selected_day, campaign=selected_campaign_ids,
                               target_col="rv_3_yr_hot_team", flag=True)
    grouped_rv1 =   filter_data(filtered_conv1, day=selected_day, campaign=selected_campaign_ids,
                               target_col="rv_3_year", flag=True)
    grouped_app  = filter_data(filtered_app,  day=selected_day, campaign=selected_campaign_ids,
                               target_col="appointment", flag=False)
    grouped_lead = filter_data(filtered_lead, day=selected_day, campaign=selected_campaign_ids,
                               target_col="Leads_created", flag=False)

    grouped_nib = filter_data(filtered_nib, day=selected_day, campaign=selected_campaign_ids,
                              target_col="not_in_buss", flag=False)
    grouped_ni = filter_data(filtered_ni, day=selected_day, campaign=selected_campaign_ids,
                             target_col="not_interested", flag=False)
    grouped_attempt = filter_data(filtered_attempt, day=selected_day, campaign=selected_campaign_ids,
                               target_col="attempt_count", flag=True)
    grouped_attempt_leads = filter_data(filtered_attempt, day=selected_day, campaign=selected_campaign_ids,
                               target_col="leads", flag=True)
    grouped_conn = filter_data(filtered_conn, day=selected_day, campaign=selected_campaign_ids,
                               target_col="connect", flag=True)
    grouped_conn_leads = filter_data(filtered_conn, day=selected_day, campaign=selected_campaign_ids,
                               target_col="leads", flag=True)
    grouped_bad_dispo = filter_data(filtered_bad_dispo, day=selected_day, campaign=selected_campaign_ids,
                              target_col="bad_dispo", flag=False)
    grouped_call_scheduling = filter_data(filtered_call_scheduling, day=selected_day, campaign=selected_campaign_ids,
                               target_col="call_scheduling_count", flag=True)
    grouped_call_scheduling_leads = filter_data(filtered_call_scheduling, day=selected_day, campaign=selected_campaign_ids,
                               target_col="leads", flag=True)

    # --------- RATE CALCULATIONS ---------
    appointment_rate = filter_appointment_rate_branch(grouped_app, grouped_lead, "appointment", "Leads_created", 'appointment_rate', True)
    nib_rate = filter_appointment_rate_branch(grouped_nib, grouped_lead, "not_in_buss", "Leads_created", 'nib_rate', True)
    ni_rate = filter_appointment_rate_branch(grouped_ni, grouped_lead, "not_interested", "Leads_created", 'ni_rate', True)
    deal_rate = filter_appointment_rate_branch(grouped_app, grouped_deal, "hot_team_leads", 'appointment', 'deal_rate', True)
    attempt_rate = filter_appointment_rate_branch(grouped_attempt, grouped_attempt_leads, "attempt_count", "leads", 'attempt_rate', True)
    connectivity_rate = filter_appointment_rate_branch(grouped_conn, grouped_conn_leads, "connect", "leads", 'conn_rate', True)
    bad_dispo_rate = filter_appointment_rate_branch(grouped_bad_dispo, grouped_lead, "bad_dispo", "Leads_created", 'bad_dispo_rate', True)
    call_scheduling_rate = filter_appointment_rate_branch(grouped_call_scheduling, grouped_call_scheduling_leads, "call_scheduling_count", "leads", 'call_scheduling_rate', True)
 # --- Pan-India summary tables ---
    # For conversion% use a slice that has both hot_team_leads & unique_leads

    conv_slice = filtered_conv.copy()
    if not conv_slice.empty:
        if 'day' in conv_slice.columns:
            conv_slice['day'] = pd.to_numeric(conv_slice['day'], errors='coerce')
            conv_slice = conv_slice[conv_slice['day'] == selected_day]
        if 'campaign_id' in conv_slice.columns and selected_campaign_ids:
            conv_slice = conv_slice[conv_slice['campaign_id'].isin(selected_campaign_ids)]

    pan_ind_con  = pan_india(conv_slice,   'hot_team_leads', True)       # conversion %
    pan_ind_deal = pan_india(grouped_deal, 'hot_team_leads', False)  
    pan_ind_deal1 = pan_india(grouped_deal1, 'deal_closed', False) 
    pan_ind_rv = pan_india(grouped_rv, "rv_3_yr_hot_team", False)  
    pan_ind_rv1 = pan_india(grouped_rv1, "rv_3_year", False)   # deals sum
    pan_ind_rv1["rv_3_year"] = (
    pan_ind_rv1["rv_3_year"]
    .apply(lambda x: '{:,.0f}'.format(x))
)
    pan_ind_app  = pan_india(grouped_app,  'appointment', False)   # appointments sum
    pan_ind_lead = pan_india(grouped_lead, 'Leads_created', False)       # leads sum

    pan_ind_deal_rate = pan_india_rate(deal_rate ,'hot_team_leads' , 'appointment' , 'deal_rate' )
    pan_ind_app_rate = pan_india_rate(appointment_rate ,'appointment' , 'Leads_created', 'appointment_rate' )
    pan_ind_nib_rate = pan_india_rate(nib_rate ,'not_in_buss' , 'Leads_created', 'nib_rate' )
    pan_ind_ni_rate = pan_india_rate(ni_rate ,'not_interested' , 'Leads_created', 'ni_rate' )
    pan_ind_attempt_rate = pan_india_rate(attempt_rate , 'attempt_count' , 'leads' , 'attempt_rate')
    pan_ind_conn_rate = pan_india_rate(connectivity_rate , 'connect' , 'leads' , 'conn_rate')
    pan_ind_bad_dispo_rate = pan_india_rate(bad_dispo_rate, 'bad_dispo', 'Leads_created', 'bad_dispo_rate')
    pan_ind_call_scheduling_rate = pan_india_rate(call_scheduling_rate, 'call_scheduling_count', 'leads', 'call_scheduling_rate')
    # --- Month formatting (no int() casts) ---
    MONTH_MAP = {1:"JAN",2:"FEB",3:"MAR",4:"APR",5:"MAY",6:"JUN",
                 7:"JUL",8:"AUG",9:"SEP",10:"OCT",11:"NOV",12:"DEC"}
    MONTH_STR_MAP = {str(i): m for i, m in MONTH_MAP.items()}
    MONTH_STR_MAP.update({f"{i:02d}": m for i, m in MONTH_MAP.items()})
    MONTH_ORDER = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]


    def _keep_last_4_months(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose month falls in the last 4 calendar months (DEC→JAN aware)."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or 'month' not in df.columns:
            return df
        out = df.copy()
        m_num = pd.to_numeric(out['month'], errors='coerce')
        if 'year' in out.columns:
            y_num = pd.to_numeric(out['year'], errors='coerce')
        else:
            cur_m = pd.Timestamp.today().month
            cur_y = pd.Timestamp.today().year
            y_num = m_num.apply(lambda m: cur_y - 1 if pd.notna(m) and m > cur_m else cur_y)
        d = pd.to_datetime(dict(year=y_num, month=m_num, day=1), errors='coerce')
        last_4 = pd.Series(d.dropna().unique()).sort_values().tail(4).tolist()
        return out.loc[d.isin(last_4)].reset_index(drop=True)

    def _format_month(df: pd.DataFrame) -> pd.DataFrame:
    # ensure df is actually a DataFrame
        if not isinstance(df, pd.DataFrame):
            try:
                # sometimes tuple like (df,) — take first element if it's DataFrame
                if isinstance(df, tuple) and isinstance(df[0], pd.DataFrame):
                    df = df[0]
                else:
                    raise TypeError("Expected a DataFrame, got {}".format(type(df)))
            except Exception:
                return pd.DataFrame()  # fail gracefully

        if df is None or df.empty or 'month' not in df.columns:
            return pd.DataFrame() if df is None or df.empty else df

        out = df.copy()
        s = out['month'].astype(str).str.strip()
        mapped = s.map(MONTH_STR_MAP)
        fallback = mapped.isna()
        if fallback.any():
            mapped.loc[fallback] = s.loc[fallback].str[:3].str.upper()
        out['month'] = mapped
        return out

    def _order_cols(df: pd.DataFrame, prefer_cols):
        if df is None or df.empty: return pd.DataFrame()
        cols = [c for c in prefer_cols if c in df.columns] + [c for c in df.columns if c not in prefer_cols]
        return df[cols]

    def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return pd.DataFrame()
        out = df.copy()
        if 'month' in out.columns:
            out['month'] = pd.Categorical(out['month'], categories=MONTH_ORDER, ordered=True)
        by = [c for c in ['month','branch'] if c in out.columns]
        out = out.sort_values(by=by, kind='stable')
        if 'month' in out.columns:
            out['month'] = out['month'].astype(str)
        return out

    def df_to_bootstrap_table(df: pd.DataFrame, num_cols: list[str] = None) -> str:
        if df is None or df.empty:
            return '<div class="text-muted small">No data for current selection.</div>'
        d = df.copy()
        num_cols = num_cols or []
        for c in num_cols:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
                # wrap in a right-align span for just the cells
                d[c] = d[c].map(lambda x: f'<span class="d-block text-end">{x:,.2f}</span>' if pd.notna(x) else '')
        try:
            html = d.to_html(index=False, classes="table table-sm table-striped table-hover align-middle mb-0",
                            border=0, escape=False, justify='left')
        except TypeError:
            html = d.to_html(index=False, classes="table table-sm table-striped table-hover align-middle mb-0",
                            border=0, escape=False).replace(
                '<tr style="text-align: right;">', '<tr style="text-align: left;">'
            )
        return html
    
    pan_ind_con  = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_con)),  ['month','branch','conversion']))
    pan_ind_deal = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_deal)), ['month','branch','hot_team_leads']))
    pan_ind_deal1 = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_deal1)), ['month','branch','deal_closed']))
    pan_ind_rv = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_rv)), ['month','branch','rv_3_yr_hot_team']))
    pan_ind_rv1 = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_rv1)), ['month','branch','rv_3_year']))
    pan_ind_app  = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_app)),  ['month','branch','appointment']))
    pan_ind_lead = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_lead)), ['month','branch','Leads_created']))
    pan_ind_app_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_app_rate)), ['month','branch','appointment_rate']))
    pan_ind_deal_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_deal_rate)), ['month','branch','deal_rate']))
    pan_ind_nib_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_nib_rate)), ['month','branch','nib_rate']))
    pan_ind_ni_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_ni_rate)), ['month','branch','ni_rate']))
    pan_ind_attempt_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_attempt_rate)), ['month','branch','attempt_rate']))
    pan_ind_conn_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_conn_rate)), ['month','branch','conn_rate']))
    pan_ind_bad_dispo_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_bad_dispo_rate)), ['month','branch','bad_dispo_rate']))
    pan_ind_call_scheduling_rate = _sort_df(_order_cols(_format_month(_keep_last_4_months(pan_ind_call_scheduling_rate)), ['month','branch','call_scheduling_rate']))
    table_conv_html = df_to_bootstrap_table(pan_ind_con)
    table_deal_html = df_to_bootstrap_table(pan_ind_deal)
    table_deal1_html = df_to_bootstrap_table(pan_ind_deal1)
    table_rv_html = df_to_bootstrap_table(pan_ind_rv)
    table_rv1_html = df_to_bootstrap_table(pan_ind_rv1)
    table_app_html  = df_to_bootstrap_table(pan_ind_app)
    table_lead_html = df_to_bootstrap_table(pan_ind_lead)
    table_app_rate_html = df_to_bootstrap_table(pan_ind_app_rate)
    table_deal_rate_html = df_to_bootstrap_table(pan_ind_deal_rate)
    table_nib_rate_html = df_to_bootstrap_table(pan_ind_nib_rate)
    table_ni_rate_html = df_to_bootstrap_table(pan_ind_ni_rate)
    table_attempt_rate_html = df_to_bootstrap_table(pan_ind_attempt_rate)
    table_conn_rate_html = df_to_bootstrap_table(pan_ind_conn_rate)
    table_bad_dispo_rate_html = df_to_bootstrap_table(pan_ind_bad_dispo_rate)
    table_call_scheduling_rate_html = df_to_bootstrap_table(pan_ind_call_scheduling_rate)
    grouped_deal.rename(columns={'hot_team_leads': 'deal_closed'}, inplace=True)
    grouped_rv.rename(columns={"rv_3_yr_hot_team": 'rv_3_year'}, inplace=True)
    # --- Charts ---
    fig_conv = plot_grouped_bar(grouped_conv, "conversion", "Conversion by Branch (Grouped by Month)")
    fig_deal = plot_grouped_bar(grouped_deal, "deal_closed", "Deals Closed by Branch (Grouped by Month)")
    fig_deal1 = plot_grouped_bar(grouped_deal1, "deal_closed", "Deals Closed by Branch(deal closed date) (Grouped by Month)")
    fig_rv = plot_grouped_bar(grouped_rv, "rv_3_year", "RV 3 Year by Branch (Grouped by Month)")
    fig_rv1 = plot_grouped_bar(grouped_rv1, "rv_3_year", "RV 3 Year by Branch(Deal closed date) (Grouped by Month)")
    fig_app  = plot_grouped_bar(grouped_app,  "appointment", "Appointments by Branch (Grouped by Month)")
    fig_lead = plot_grouped_bar(grouped_lead, "Leads_created", "Leads by Branch (Grouped by Month)")
    fig_app_rate = plot_grouped_bar(appointment_rate, "appointment_rate", "Leads to Appointment Branch (Grouped by Month)")
    fig_deal_rate = plot_grouped_bar(deal_rate, "deal_rate", "Appointment to deal_closed by Branch (Grouped by Month)")
    fig_nib_rate = plot_grouped_bar(nib_rate, "nib_rate", "not in business rate by Branch (Grouped by Month)")
    fig_ni_rate = plot_grouped_bar(ni_rate, "ni_rate", "not interested rate by Branch (Grouped by Month)")
    fig_attempt_rate = plot_grouped_bar(attempt_rate, "attempt_rate", "5+ Attempt rate by Branch (Grouped by Month)")
    fig_conn_rate = plot_grouped_bar(connectivity_rate, "conn_rate", "Connectivity Rate by Branch (Grouped by Month)")
    fig_bad_dispo_rate = plot_grouped_bar(bad_dispo_rate, "bad_dispo_rate", "Bad Disposition Rate by Branch (Grouped by Month)")
    fig_call_scheduling_rate = plot_grouped_bar(call_scheduling_rate, "call_scheduling_rate", "Call Scheduling Rate by Branch (Grouped by Month)")

    html = render_template(
        'branch.html',
        page="branch",
        refreshed=is_update_click,
        last_updated=last_updated,
        campaign_choices=campaign_name_choices,
        campaigns_selected=selected_campaign_names,
        # campaigns_selected=selected_campaign_ids,
        selected_day=selected_day,
        # campaign_choices=campaign_name_choices,
        graph_conv=fig_conv.to_html(full_html=False, include_plotlyjs=False),
        graph_deal=fig_deal.to_html(full_html=False, include_plotlyjs=False),
        graph_deal1=fig_deal1.to_html(full_html=False, include_plotlyjs=False),
        graph_rv=fig_rv.to_html(full_html=False, include_plotlyjs='cdn'),
        graph_rv1=fig_rv1.to_html(full_html=False, include_plotlyjs=False),
        graph_app=fig_app.to_html(full_html=False, include_plotlyjs=False),
        graph_lead=fig_lead.to_html(full_html=False, include_plotlyjs=False),
        graph_app_rate=fig_app_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_deal_rate=fig_deal_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_nib_rate=fig_nib_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_ni_rate=fig_ni_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_attempt_rate=fig_attempt_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_conn_rate=fig_conn_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_bad_dispo_rate=fig_bad_dispo_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_call_scheduling_rate=fig_call_scheduling_rate.to_html(full_html=False, include_plotlyjs=False),
        table_conv=table_conv_html,
        table_deal=table_deal_html,
        table_deal1=table_deal1_html,
        table_rv=table_rv_html,
        table_rv1=table_rv1_html,
        table_app=table_app_html,
        table_lead=table_lead_html,
        table_app_rate=table_app_rate_html,
        table_deal_rate=table_deal_rate_html,
        table_nib_rate=table_nib_rate_html,
        table_ni_rate=table_ni_rate_html,
        table_attempt_rate=table_attempt_rate_html,
        table_conn_rate=table_conn_rate_html,
        table_bad_dispo_rate=table_bad_dispo_rate_html,
        table_call_scheduling_rate=table_call_scheduling_rate_html,
    )

    resp = make_response(html)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@application.route('/', methods=['GET', 'POST'])
def data_dashboard():

    # -------------------------------------------------
    # UPDATE DATA FLAG
    # -------------------------------------------------
    update_clicked = request.method == 'POST' and request.form.get('update_data') == '1'
    dfs = _get_data(force_update=update_clicked)
    last_updated = _DATA_CACHE["fetched_at"].strftime("%d-%b-%Y %I:%M %p") if _DATA_CACHE["fetched_at"] else "Never"

    appointment_data     = dfs['appointment']
    lead_generation_data = dfs['lead']
    conversion_data      = dfs['conversion']
    conversion_data1     = dfs['conversion_deal']
    NIB_data             = dfs['NIB']
    NI_data              = dfs['NI']
    attempt_data         = dfs['attempt']
    conn_data            = dfs['connectivity']
    tme_data            = dfs['tme_experience']
    bad_dispo_data      = dfs['bad_dispo']
    call_scheduling_data = dfs['call_scheduling']

    # =================================================
    # MARKETING CAMPAIGN (NAME → ID)
    # =================================================
    campaign_name_choices = sorted(dict_camp.keys())

    def campaign_names_to_ids(names):
        return [dict_camp[n] for n in names if n in dict_camp]

    selected_campaign_names = request.form.getlist('campaign_id')

    campaigns_selected = (
    campaign_names_to_ids(selected_campaign_names)
    if selected_campaign_names
    else list(dict_camp.values())   # ALL marketing campaigns by default
    )

    # =================================================
    # PRODUCT CAMPAIGN (NAME → ID)
    # =================================================
    product_campaign_name_choices = sorted(dict_camp_product.keys())

    def product_campaign_names_to_ids(names):
        return [dict_camp_product[n] for n in names if n in dict_camp_product]

    selected_product_campaign_names = request.form.getlist('product_campaign_id')

    product_campaigns_selected = (
        product_campaign_names_to_ids(selected_product_campaign_names)
        if selected_product_campaign_names else []
    )

    # =================================================
    # GLOBAL CAMPAIGN IDS (EMPTY OK)
    # =================================================
    raw_campaigns = campaigns_selected + product_campaigns_selected

    global_campaigns_selected = [
        int(c) for c in raw_campaigns
        if isinstance(c, (int, np.integer)) or
           (isinstance(c, str) and c.isdigit())
    ]

    # =================================================
    # MONTH / BRANCH (NO DEFAULT AUTO-SELECTION)
    # =================================================
    campaign_choices, month_choices_raw, branch_choices = _choices_from_data(dfs)

    MONTH_NAME_MAP = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    MONTH_NAME_REVERSE = {v: k for k, v in MONTH_NAME_MAP.items()}

    # Sort months chronologically (handles year wrap) and take last 4
    current_month = datetime.now().month
    month_choices_sorted = sorted(
        month_choices_raw,
        key=lambda m: (m - current_month - 1) % 12
    )
    month_choices_last4 = month_choices_sorted[-4:] if len(month_choices_sorted) > 4 else month_choices_sorted

    # Convert to name strings for UI
    month_choices = [MONTH_NAME_MAP.get(m, str(m)) for m in month_choices_last4]

    # Read selected month names from form and convert back to ints
    # month_names_selected = request.values.getlist('month')
    # months_selected = [MONTH_NAME_REVERSE[n] for n in month_names_selected if n in MONTH_NAME_REVERSE]
    month_names_selected = request.values.getlist('month')

# If nothing selected -> all months
    if month_names_selected:
        months_selected = [
            MONTH_NAME_REVERSE[n]
            for n in month_names_selected
            if n in MONTH_NAME_REVERSE
        ]
    else:
        months_selected = month_choices_last4[:]   # last 4 months
        month_names_selected = [
            MONTH_NAME_MAP[m]
            for m in month_choices_last4
        ]
    branches_selected = request.values.getlist('branch')

    # IMPORTANT: empty list means "no filter"
    branches_param = branches_selected or []

    # =================================================
    # DISPLAY (UI ONLY)
    # =================================================
    selected_campaigns_display = (
        f"Marketing: {', '.join(selected_campaign_names) if selected_campaign_names else 'None'} | "
        f"Product: {', '.join(selected_product_campaign_names) if selected_product_campaign_names else 'None'}"
    )

    selected_months_display = ", ".join(month_names_selected) if month_names_selected else "None"
    selected_branches_display = ", ".join(branches_selected) if branches_selected else "None"

    # =================================================
    # FILTER DATA (EMPTY LIST SAFE)
    # =================================================
    filtered_conv = filter_conversion_data(
        conversion_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_conv1 = filter_conversion_data1(
        conversion_data1, global_campaigns_selected, months_selected, branches_param
    )

    filtered_lead = filter_lead_generation_data(
        lead_generation_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_app = filter_appointment_data(
        appointment_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_app_rate = filter_appointment_rate(
        filtered_app, filtered_lead, 'Leads_created_cum', 'appointment_rate', True
    )

    filtered_deal_rate = filter_appointment_rate(
        filtered_app, filtered_conv, 'hot_team_leads', 'deal_rate', False
    )

    filtered_nib = filter_NIB_data(
        NIB_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_nib_rate = filter_nib_rate(
        filtered_nib, filtered_lead, 'Leads_created_cum', 'NIB_Rate'
    )

    filtered_ni = filter_ni_data(
        NI_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_ni_rate = filter_ni_rate(
        filtered_ni, filtered_lead, 'Leads_created_cum', 'NI_Rate'
    )

    filtered_bad_dispo = filter_bad_dispo_data(
        bad_dispo_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_bad_dispo_rate = filter_bad_dispo_rate(
        filtered_bad_dispo, filtered_lead, 'Leads_created_cum', 'Bad_Dispo_Rate'
    )

    filtered_attempt = filter_attempt_data(
        attempt_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_rv_lead = filter_rv_data(
        conversion_data, global_campaigns_selected, months_selected, branches_param)
    

    filtered_attempt_rate = filter_attempt_rate(
        filtered_attempt, 'attempt_rate'
    )
    filtered_rv_lead_rate = filter_rv_rate(
        filtered_rv_lead, 'RV_Lead_Rate' )

    filtered_call_scheduling = filter_call_scheduling_data(
        call_scheduling_data, global_campaigns_selected, months_selected, branches_param
    )

    filtered_call_scheduling_rate = filter_call_scheduling_rate(
        filtered_call_scheduling, 'call_scheduling_rate'
    )

    filtered_conn = filter_connectivity(
        conn_data, global_campaigns_selected, months_selected, branches_param
    )
    filtered_tme = filter_tme_experience(
        tme_data, global_campaigns_selected, months_selected, branches_param)
    # =================================================
    # PREP FOR PLOTTING
    # =================================================
    filtered_conv = _prepare_df(filtered_conv)
    filtered_conv1 = _prepare_df(filtered_conv1)
    filtered_lead = _prepare_df(filtered_lead)
    filtered_app  = _prepare_df(filtered_app)
    filtered_app_rate = _prepare_df(filtered_app_rate)
    filtered_deal_rate = _prepare_df(filtered_deal_rate)
    filtered_nib_rate = _prepare_df(filtered_nib_rate)
    filtered_ni_rate = _prepare_df(filtered_ni_rate)
    filtered_bad_dispo_rate = _prepare_df(filtered_bad_dispo_rate)
    filtered_attempt_rate = _prepare_df(filtered_attempt_rate)
    filtered_rv_lead_rate = _prepare_df(filtered_rv_lead_rate)
    filtered_call_scheduling_rate = _prepare_df(filtered_call_scheduling_rate)
    filtered_connectivity = _prepare_df(filtered_conn)
    filtered_tme = _prepare_df(filtered_tme)

    # =================================================
    # CHARTS
    # =================================================
    fig_conv = make_timeseries_figure(filtered_conv, months_selected, 'conversion', "Conversion Over Time", "Conversion")
    fig_lead = make_timeseries_figure(filtered_lead, months_selected, 'Leads_created_cum', "Leads Over Time", "Leads")
    fig_app  = make_timeseries_figure(filtered_app, months_selected, 'appointment', "Appointments Over Time", "Appointments")
    fig_deal = make_timeseries_figure(filtered_conv, months_selected, 'hot_team_leads', "Deals Closed Over Time", "Deals Closed")
    fig_deal1 = make_timeseries_figure(filtered_conv1, months_selected, 'deal_closed', "Deals Closed Over Time (Deal Closed Date)", "Deals Closed")
    fig_rv = make_timeseries_figure(filtered_conv, months_selected, 'rv_3_yr_hot_team', "RV 3 Year Over Time", "RV 3 Year")
    fig_rv1 = make_timeseries_figure(filtered_conv1, months_selected, 'rv_3_year', "RV 3 Year Over Time (Deal Closed Date)", "RV 3 Year")
    fig_app_rate  = make_timeseries_figure(filtered_app_rate, months_selected, 'appointment_rate', "Leads → Appointment Rate", "Appointment Rate")
    fig_deal_rate = make_timeseries_figure(filtered_deal_rate, months_selected, 'deal_rate', "Appointment → Deal Rate", "Deal Rate")
    fig_nib = make_timeseries_figure(filtered_nib_rate, months_selected, 'NIB_Rate', "NIB Rate", "NIB Rate")
    fig_ni  = make_timeseries_figure(filtered_ni_rate, months_selected, 'NI_Rate', "NI Rate", "NI Rate")
    fig_bad_dispo = make_timeseries_figure(filtered_bad_dispo_rate, months_selected, 'Bad_Dispo_Rate', "Bad Disposition Rate", "Bad Disposition Rate")
    fig_rv_lead_rate = make_timeseries_figure(filtered_rv_lead_rate, months_selected, 'RV_Lead_Rate', "3 Year RV Per Lead", "3 Year RV Per Lead")
    fig_attempt = make_timeseries_figure(filtered_attempt_rate, months_selected, 'attempt_rate', "Attempt Rate", "Attempt Rate")
    fig_call_scheduling = make_timeseries_figure(filtered_call_scheduling_rate, months_selected, 'call_scheduling_rate', "Call Scheduling Rate", "Call Scheduling Rate")
    fig_conn = make_timeseries_figure(filtered_connectivity, months_selected, 'conn_per', "Connectivity Rate", "Connectivity Rate")
    fig_tme = make_timeseries_figure(filtered_tme, months_selected, 'avg_experience', "TME Average Experience (Days)", "Avg Experience (Days)")
    fig_tme_no = make_timeseries_figure(filtered_tme, months_selected, 'unique_tme', "Number of TME", "Number of TME")
    # fig_tme_linebar = make_monthly_line_bar(filtered_tme, months_selected, 'total_emp', 'tme_experience_rate', "TME Experience vs No. of Employees")
    # =================================================
    # RENDER
    # =================================================
    html = render_template(
        'dashboard.html',
        refreshed=update_clicked,
        last_updated=last_updated,

        campaign_choices=campaign_name_choices,
        campaigns_selected=selected_campaign_names,

        product_campaign_choices=product_campaign_name_choices,
        product_campaigns_selected=selected_product_campaign_names,

        month_choices=month_choices,
        month_selected=month_names_selected,

        branch_choices=branch_choices,
        branches_selected=branches_selected,

        selected_campaigns=selected_campaigns_display,
        selected_months_display=selected_months_display,
        selected_branches_display=selected_branches_display,

        graph_conv=fig_conv.to_html(full_html=False, include_plotlyjs=False),
        graph_lead=fig_lead.to_html(full_html=False, include_plotlyjs=False),
        graph_app=fig_app.to_html(full_html=False, include_plotlyjs=False),
        graph_deal=fig_deal.to_html(full_html=False, include_plotlyjs=False),
        graph_deal1=fig_deal1.to_html(full_html=False, include_plotlyjs=False),
        graph_rv=fig_rv.to_html(full_html=False, include_plotlyjs='cdn'),
        graph_rv1=fig_rv1.to_html(full_html=False, include_plotlyjs=False),
        graph_app_rate=fig_app_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_deal_rate=fig_deal_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_nib=fig_nib.to_html(full_html=False, include_plotlyjs=False),
        graph_ni=fig_ni.to_html(full_html=False, include_plotlyjs=False),
        graph_tme=fig_tme.to_html(full_html=False, include_plotlyjs=False),
        graph_tme_no=fig_tme_no.to_html(full_html=False, include_plotlyjs=False),
        graph_attempt=fig_attempt.to_html(full_html=False, include_plotlyjs=False),
        graph_rv_lead_rate=fig_rv_lead_rate.to_html(full_html=False, include_plotlyjs=False),
        graph_call_scheduling=fig_call_scheduling.to_html(full_html=False, include_plotlyjs=False),
        graph_conn=fig_conn.to_html(full_html=False, include_plotlyjs=False),
        graph_bad_dispo=fig_bad_dispo.to_html(full_html=False, include_plotlyjs=False),
    )

    resp = make_response(html)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp
