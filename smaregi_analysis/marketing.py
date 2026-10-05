"""Weekly brand/marketing KPIs (認知KPI_週次トラッカー) set against store (Smaregi) sales.

Inputs are two CSVs exported from the tracker sheet, one row per Sunday-start week:
  tracker.csv : week, gsc_impr, gsc_clicks, gsc_ctr, gsc_pos, ig_reach, ig_impr, sessions, followers, ig_sessions
  activity.csv: week, ig_posts, newsletters, social_sessions, email_sessions, sessions, checkouts, conv_rate,
                sales_incl_store   (Shopify total incl. the store, as in the 「活動→効果」 sheet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_weekly(tracker_path: str, activity_path: str | None = None) -> pd.DataFrame:
    t = pd.read_csv(tracker_path, parse_dates=["week"])
    if activity_path:
        a = pd.read_csv(activity_path, parse_dates=["week"]).drop(columns=["sessions"], errors="ignore")
        t = t.merge(a, on="week", how="left")
    return t.sort_values("week").reset_index(drop=True)


def _rho(x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    if ok.sum() < 6:
        return dict(rho=None, p=None, n=int(ok.sum()))
    r = spearmanr(x[ok], y[ok])
    return dict(rho=float(r.statistic), p=float(r.pvalue), n=int(ok.sum()))


# Columns that come from Shopify and follow a partial week's measured days (`from`/`to`);
# search impressions, reach and followers are always full-week figures.
PARTIAL_COLS = {"sessions", "ig_sessions", "social_sessions", "email_sessions", "checkouts", "sales_incl_store"}


def _to_months(w: pd.DataFrame, cols: list[str], min_cover: float = 0.9) -> pd.DataFrame:
    """Spread each week evenly over its days and total by calendar month.

    Shopify columns use a partial week's measured days (`from`/`to`). A month whose days are
    not all covered is scaled up from the covered days, or dropped below `min_cover`.
    """
    rows = []
    for r in w.to_dict("records"):
        full = pd.date_range(r["week"], periods=7)
        start = pd.Timestamp(r["from"]) if pd.notna(r.get("from")) else full[0]
        stop = pd.Timestamp(r["to"]) if pd.notna(r.get("to")) else full[-1]
        part = pd.date_range(start, stop)
        for c in cols:
            if pd.isna(r[c]):
                continue
            days = part if c in PARTIAL_COLS else full
            rows += [(d, c, r[c] / len(days)) for d in days]
    if not rows:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(rows, columns=["day", "col", "v"])
    d["month"] = d["day"].dt.to_period("M")
    total = d.pivot_table(index="month", columns="col", values="v", aggfunc="sum")
    cover = d.groupby(["month", "col"])["day"].nunique().unstack() / total.index.days_in_month.to_numpy()[:, None]
    out = (total / cover).where(cover >= min_cover)
    return out.reindex(columns=cols)


def analyze(weekly: pd.DataFrame, monthly: pd.DataFrame, styles: pd.DataFrame, end: str) -> dict:
    """Facts for the report. Weeks that are not complete by `end` are left out (and reported)."""
    end_ts = pd.Timestamp(end)
    w = weekly.copy()
    w["complete"] = w["week"] + pd.Timedelta(days=6) <= end_ts
    incomplete = [str(d.date()) for d in w.loc[~w.complete, "week"]]
    w = w[w.complete].copy()
    w["ig_rate"] = w["ig_sessions"] / w["ig_reach"]
    w["net_follow"] = w["followers"].diff()
    # Partially measured weeks (marked with from/to, 'not for comparison' in the sheet) stay out of
    # week-to-week comparisons.
    partial = w["from"].notna() | w["to"].notna() if "from" in w else pd.Series(False, index=w.index)
    cmp = w[~partial]

    # --- Instagram: does reach turn into visits?
    med_reach = float(cmp["ig_reach"].median())
    high = cmp[cmp["ig_reach"] >= 2 * med_reach]
    rest = cmp[cmp["ig_reach"] < 2 * med_reach]
    rest_rate = float(rest.ig_sessions.sum() / rest.ig_reach.sum()) if len(rest) else np.nan
    spikes = [dict(week=str(r.week.date()), reach=int(r.ig_reach), ig_sessions=int(r.ig_sessions), rate=float(r.ig_rate),
                   checkouts=None if pd.isna(getattr(r, "checkouts", np.nan)) else int(r.checkouts),
                   net_follow=None if pd.isna(r.net_follow) else int(r.net_follow),
                   converting=bool(r.ig_rate >= rest_rate)) for r in high.itertuples()]
    conv = high[high.ig_rate >= rest_rate]
    flat = high[high.ig_rate < rest_rate]
    ig = dict(median_reach=med_reach, rest_rate=rest_rate, spikes=spikes,
              conv_rate=float(conv.ig_sessions.sum() / conv.ig_reach.sum()) if len(conv) else None,
              flat_rate=float(flat.ig_sessions.sum() / flat.ig_reach.sum()) if len(flat) else None,
              conv_reach=int(conv.ig_reach.sum()), flat_reach=int(flat.ig_reach.sum()),
              conv_sessions=int(conv.ig_sessions.sum()), flat_sessions=int(flat.ig_sessions.sum()),
              conv_months=sorted({f"{d.month}月" for d in conv.week}, key=lambda x: int(x[:-1])),
              flat_months=sorted({f"{d.month}月" for d in flat.week}, key=lambda x: int(x[:-1])),
              best=[dict(week=str(r.week.date()), reach=int(r.ig_reach), ig_sessions=int(r.ig_sessions), rate=float(r.ig_rate))
                    for r in cmp.nlargest(4, "ig_rate").itertuples()])

    # --- online orders: what moves them
    have_orders = cmp.dropna(subset=["checkouts"]) if "checkouts" in cmp else cmp.iloc[0:0]
    if len(have_orders):
        med_o = have_orders["checkouts"].median()
        peak_mask = have_orders["checkouts"] >= 3 * med_o
        peak_share = float(have_orders.loc[peak_mask, "checkouts"].sum() / have_orders["checkouts"].sum())
        top_email = have_orders.loc[have_orders["email_sessions"].idxmax()]
        no_jan = have_orders[have_orders.week.dt.month != 1]
    orders = dict(
        weeks=int(len(have_orders)), median=float(have_orders["checkouts"].median()) if len(have_orders) else None,
        top=[dict(week=str(r.week.date()), checkouts=int(r.checkouts), email=int(r.email_sessions), social=int(r.social_sessions),
                  reach=int(r.ig_reach)) for r in have_orders.nlargest(4, "checkouts").itertuples()],
        vs_email=_rho(have_orders["email_sessions"], have_orders["checkouts"]),
        vs_email_prev_week=_rho(have_orders["email_sessions"].shift(1), have_orders["checkouts"]),
        vs_social=_rho(have_orders["social_sessions"], have_orders["checkouts"]),
        vs_reach=_rho(have_orders["ig_reach"], have_orders["checkouts"]),
        vs_email_no_jan=_rho(no_jan["email_sessions"], no_jan["checkouts"]),
        peak_weeks=int(peak_mask.sum()), peak_share=peak_share,
        top_email=dict(week=str(top_email.week.date()), email=int(top_email.email_sessions), checkouts=int(top_email.checkouts)),
    ) if len(have_orders) else None

    # --- followers
    last = w.iloc[-1]
    recent = w[w.week >= last.week - pd.Timedelta(weeks=13)]
    pace = float((recent.followers.iloc[-1] - recent.followers.iloc[0]) / max(len(recent) - 1, 1))
    year_end = pd.Timestamp(f"{last.week.year}-12-26")
    weeks_left = max((year_end - last.week).days / 7, 0)
    follow = dict(now=int(last.followers), week=str(last.week.date()), pace=pace,
                  projected=float(last.followers + pace * weeks_left), weeks_left=weeks_left,
                  needed=float((20000 - last.followers) / weeks_left) if weeks_left else None)

    # --- brand search: quarter averages
    q = cmp.assign(q=cmp.week.dt.quarter).groupby("q").agg(impr=("gsc_impr", "mean"), clicks=("gsc_clicks", "mean"),
                                                         pos=("gsc_pos", "mean"), ctr=("gsc_ctr", "mean"))
    search = [dict(q=int(k), **{c: float(v) for c, v in r.items()}) for k, r in q.iterrows()]

    # --- store vs funnel, monthly
    mm = _to_months(w, ["gsc_impr", "ig_reach", "sessions", "ig_sessions"])
    mon = monthly.set_index("month")
    prev = mon.copy()
    prev.index = prev.index + 12
    store = mon[["純売上(税抜)", "取引数", "販売点数", "客単価"]].join(prev[["取引数", "販売点数", "純売上(税抜)", "客単価"]].add_suffix("_prev"))
    full_months = [p for p in mm.index if p.end_time <= end_ts and p in store.index]
    j = mm.loc[full_months].join(store)
    j["tx_yoy"] = j["取引数"] / j["取引数_prev"] - 1
    j = j.dropna(subset=["gsc_impr", "sessions"])
    store_corr = {c: _rho(j[c], j["取引数"]) for c in ["gsc_impr", "ig_reach", "sessions", "ig_sessions"]}
    season_rho = _rho(j["取引数"], j["取引数_prev"])
    dutyfree = (mon["免税額"] * 10 / mon["純売上(税抜)"]).loc[[p for p in j.index]]
    corr_months = [str(p) for p in j.index]
    recent_m = j.tail(3)
    store_recent = [dict(month=str(p), tx_yoy=float(r.tx_yoy), units_yoy=float(r["販売点数"] / r["販売点数_prev"] - 1),
                         aov_yoy=float(r["客単価"] / r["客単価_prev"] - 1),
                         unit_price_yoy=float((r["純売上(税抜)"] / r["販売点数"]) / (r["純売上(税抜)_prev"] / r["販売点数_prev"]) - 1))
                    for p, r in recent_m.iterrows()]

    # Same products, same year-on-year price? (mix vs price)
    o = styles[(styles.grp != "他社") & (styles.status == "both") & (styles.u25 >= 10) & (styles.u26 >= 10)]
    like_for_like = float(np.average((o.s26 / o.u26) / (o.s25 / o.u25), weights=o.u26)) if len(o) else None

    # --- online sales outside Smaregi: Shopify total incl. store minus Smaregi 総売上, by month
    online = []
    if "sales_incl_store" in w:
        ms = _to_months(w.dropna(subset=["sales_incl_store"]), ["sales_incl_store", "checkouts"], min_cover=0.999)
        for p, r in ms.dropna().iterrows():
            if p.end_time > end_ts or p not in mon.index or r.sales_incl_store <= 0:
                continue
            # Which Smaregi measure matches Shopify's total is not documented, so show both:
            # 総売上 adds back the waived duty-free tax and discounts, 純売上 is what was charged.
            lo = r.sales_incl_store - mon.loc[p, "総売上"]
            hi = r.sales_incl_store - mon.loc[p, "純売上"]
            online.append(dict(month=str(p), total=float(r.sales_incl_store), share_low=float(lo / r.sales_incl_store),
                               share_high=float(hi / r.sales_incl_store), checkouts=float(r.checkouts)))

    # IG-sourced visits are a subset of all social visits by the sheet's definition.
    ig_inconsistent = []
    if "social_sessions" in w:
        bad = w[(w.social_sessions.notna()) & (w.ig_sessions >= w.social_sessions) & (w.ig_sessions >= 50)]
        ig_inconsistent = [str(d.date()) for d in bad.week]
    charts = w[["week", "gsc_impr", "gsc_pos", "ig_reach", "ig_sessions", "sessions", "followers"]
               + [c for c in ["checkouts", "email_sessions", "social_sessions", "sales_incl_store"] if c in w]].copy()
    charts["week"] = charts["week"].dt.strftime("%Y-%m-%d")
    return dict(incomplete_weeks=incomplete, ig=ig, orders=orders, follow=follow, search=search,
                store_corr=store_corr, corr_months=corr_months, season_rho=season_rho,
                dutyfree_min=float(dutyfree.min()), dutyfree_max=float(dutyfree.max()),
                ig_inconsistent=ig_inconsistent, store_recent=store_recent, like_for_like_price=like_for_like,
                online=online, weekly=charts.replace({np.nan: None}).to_dict("records"),
                first_week=str(w.week.iloc[0].date()), last_week=str(w.week.iloc[-1].date()))
