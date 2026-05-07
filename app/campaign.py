from flask import render_template, request, make_response
import pandas as pd
import numpy as np
from app.campaign_utilis import filter_data, plot_grouped_bar, filter_data_conv, filter_appointment_rate_branch


def campaign_analysis():
    """
    Two forms on this page:
      1) Top 'Update Data' form -> posts 'update_data=1'
      2) Filters form (branch/day) -> posts 'branch', 'day'
    """
    # Lazy import to avoid circular dependency (routes.py imports campaign.py)
    from app.routes import _get_data, _choices_from_data, _max_day_across, _DATA_CACHE

    dict_map = {
        111: "pmax",
        112: "pmax",
        108: "fb",
        159: "fb",
        196: "fb",
        285: "fb",
        286: "fb",
        160: "wa_marketing",
        200: "wa_marketing",
        256: "wa_seasonal_spike",
        258: "competitor_fomo",
        259: "competitor_fomo",
        208: "category_surge_super",
        242: "seasonal_super",
        198: "seasonal_super",
        199: "seasonal_super",
        260: "offer_notification",
        243: "wa_pilot_a",
        288: "marketing_email_super"
    }

    campaign_id_list = [
        111, 112,
        108, 159, 196, 285, 286,
        160, 200,
        256,
        258, 259,
        208,
        242, 198, 199,
        260,
        243, 288
    ]

    is_update_click = request.method == 'POST' and request.form.get('update_data') == '1'
    is_filters_click = request.method == 'POST' and not is_update_click

    # Shared data cache with dashboard and branch pages
    dfs = _get_data(force_update=is_update_click)
    last_updated = _DATA_CACHE["fetched_at"].strftime("%d-%b-%Y %I:%M %p") if _DATA_CACHE["fetched_at"] else "Never"

    campaign_choices, month_choices, branch_choices = _choices_from_data(dfs)

    appointment_df = dfs['appointment']
    lead_df        = dfs['lead']
    conversion_df  = dfs['conversion']
    conversion_df1 = dfs['conversion_deal']
    NIB_data       = dfs['NIB']
    NI_data        = dfs['NI']
    attempt        = dfs['attempt']
    bad_dispo_df   = dfs['bad_dispo']

    default_day = _max_day_across(dfs)
    if is_filters_click:
        day_raw = request.form.get('day')
        branches_selected = request.values.getlist('branch')
    else:
        day_raw = None
        branches_selected = []  # default to NONE – user must select
    try:
        selected_day = int(float(day_raw)) if day_raw else default_day
    except (ValueError, TypeError):
        selected_day = default_day

    branches_param = branches_selected  # empty list = no data shown
    selected_branches_display = ", ".join(branches_selected) if branches_selected else "None"

    filtered_conv = conversion_df.copy()
    filtered_conv1 = conversion_df1.copy()
    filtered_lead = lead_df.copy()
    filtered_app  = appointment_df.copy()
    filtered_nib  = NIB_data.copy()
    filtered_ni   = NI_data.copy()
    filtered_attempt = attempt.copy()
    filtered_bad_dispo = bad_dispo_df.copy()
    for df in (
        filtered_conv, filtered_conv1, filtered_lead, filtered_app,
        filtered_nib, filtered_ni, filtered_attempt, filtered_bad_dispo
    ):
        if not df.empty and 'campaign_id' in df.columns:
            df['campaign_id'] = pd.to_numeric(
                df['campaign_id'], errors='coerce'
            ).astype('Int64')

            mask = df['campaign_id'].isin(campaign_id_list)
            df.drop(df.index[~mask], inplace=True)

            df['campaign_name'] = df['campaign_id'].map(dict_map)

    filtered_app.rename(columns={'appointment_close': 'appointment'}, inplace=True)

    grouped_conv = filter_data_conv(filtered_conv, selected_day, branches_param)
    grouped_deal = filter_data(filtered_conv, day=selected_day, branch=branches_param,
                               target_col="hot_team_leads", flag=True)
    grouped_deal1 = filter_data(filtered_conv1, day=selected_day, branch=branches_param,
                               target_col="deal_closed", flag=True)
    grouped_rv = filter_data(filtered_conv, day=selected_day, branch=branches_param,
                               target_col="rv_3_yr_hot_team", flag=True)
    grouped_rv1 = filter_data(filtered_conv1, day=selected_day, branch=branches_param,
                               target_col="rv_3_year", flag=True)
    grouped_app = filter_data(filtered_app, day=selected_day, branch=branches_param,
                               target_col="appointment", flag=False)
    grouped_lead = filter_data(filtered_lead, day=selected_day, branch=branches_param,
                               target_col="Leads_created", flag=False)
    grouped_nib = filter_data(filtered_nib, day=selected_day, branch=branches_param,
                              target_col="not_in_buss", flag=False)
    grouped_ni = filter_data(filtered_ni, day=selected_day, branch=branches_param,
                             target_col="not_interested", flag=False)
    grouped_attempt = filter_data(filtered_attempt, day=selected_day, branch=branches_param,
                               target_col="attempt_count", flag=False)
    grouped_attempt_leads = filter_data(filtered_attempt, day=selected_day, branch=branches_param,
                               target_col="leads", flag=False)
    grouped_bad_dispo = filter_data(filtered_bad_dispo, day=selected_day, branch=branches_param,
                              target_col="bad_dispo", flag=False)

    appointment_rate = filter_appointment_rate_branch(grouped_app, grouped_lead, "appointment", "Leads_created", 'appointment_rate', True)
    nib_rate = filter_appointment_rate_branch(grouped_nib, grouped_lead, "not_in_buss", "Leads_created", 'nib_rate', True)
    ni_rate = filter_appointment_rate_branch(grouped_ni, grouped_lead, "not_interested", "Leads_created", 'ni_rate', True)
    deal_rate = filter_appointment_rate_branch(grouped_app, grouped_deal, "hot_team_leads", 'appointment', 'deal_rate', False)
    attempt_rate = filter_appointment_rate_branch(grouped_attempt, grouped_attempt_leads, "attempt_count", "leads", 'attempt_rate', True)
    bad_dispo_rate = filter_appointment_rate_branch(grouped_bad_dispo, grouped_lead, "bad_dispo", "Leads_created", 'bad_dispo_rate', True)
    # --- Charts ---
    grouped_deal.rename(columns={"hot_team_leads": 'Deal_closed'}, inplace=True)
    fig_conv = plot_grouped_bar(grouped_conv, "conversion", "Conversion by Campaign (Grouped by Month)")
    fig_deal = plot_grouped_bar(grouped_deal, 'Deal_closed', "Deals Closed by Campaign (Grouped by Month)")
    fig_deal1 = plot_grouped_bar(grouped_deal1, 'deal_closed', "Deals Closed by Campaign (Deal closed date) (Grouped by Month)")
    fig_rv = plot_grouped_bar(grouped_rv, "rv_3_yr_hot_team", "RV 3 Year by Campaign (Grouped by Month)")
    fig_rv1 = plot_grouped_bar(grouped_rv1, "rv_3_year", "RV 3 Year by Campaign (Deal closed date) (Grouped by Month)")
    fig_app = plot_grouped_bar(grouped_app, "appointment", "Appointments by Campaign (Grouped by Month)")
    fig_lead = plot_grouped_bar(grouped_lead, "Leads_created", "Leads by Campaign (Grouped by Month)")
    fig_app_rate = plot_grouped_bar(appointment_rate, "appointment_rate", "Leads to Appointment Rate by Campaign (Grouped by Month)")
    fig_deal_rate = plot_grouped_bar(deal_rate, "deal_rate", "Appointment to Deal Rate by Campaign (Grouped by Month)")
    fig_nib_rate = plot_grouped_bar(nib_rate, "nib_rate", "NIB Rate by Campaign (Grouped by Month)")
    fig_ni_rate = plot_grouped_bar(ni_rate, "ni_rate", "Not Interested Rate by Campaign (Grouped by Month)")
    fig_attempt_rate = plot_grouped_bar(attempt_rate, "attempt_rate", "5+ Attempt Rate by Campaign (Grouped by Month)")
    fig_bad_dispo_rate = plot_grouped_bar(bad_dispo_rate, "bad_dispo_rate", "Bad Disposition Rate by Campaign (Grouped by Month)")
    html = render_template(
        'campaign.html',
        page="campaign",
        refreshed=is_update_click,
        last_updated=last_updated,
        selected_day=selected_day,
        branch_choices=branch_choices,
        branches_selected=branches_selected,
        selected_branches_display=selected_branches_display,
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
        graph_bad_dispo_rate=fig_bad_dispo_rate.to_html(full_html=False, include_plotlyjs=False),
    )

    resp = make_response(html)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp
