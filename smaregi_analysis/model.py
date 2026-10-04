"""Stop-selling break-point logic per style (品番).

Break-point: a style should stop being sold once the gross profit it is expected to earn
falls below what a typical NEW product earns on the same shelf (median first-period gross
profit of this year's launches, H50). Stock-outs, launch timing and autumn/winter seasonality
are corrected before the comparison, and every result carries the reason and a stability score.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.special import betainc

from . import taxonomy as tx
from .loaders import normalize_color
from .params import Calendar, implied_sellout_date

STOP_LABELS = ("終売（在庫消化）", "終売済み")
LABEL_ORDER = ["再生産", "再生産・復刻（要確認）", "継続・強化", "アップデート", "様子見",
               "終売（在庫消化）", "終売済み", "他社ブランド別管理"]
TOPS = ["Tシャツ", "スウェット・フーディ", "キッズ"]


def hd_quantile(x, p: float) -> float:
    """Harrell–Davis quantile: a Beta-weighted mean of all order statistics, steadier than the sample quantile."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    a, b = (n + 1) * p, (n + 1) * (1 - p)
    w = np.diff(betainc(a, b, np.arange(n + 1) / n))
    return float(np.dot(w, x))


# --------------------------------------------------------------------------------------------
# SKU table
# --------------------------------------------------------------------------------------------
def build_sku_table(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU with previous-year and current-window figures side by side."""
    def per_sku(d: pd.DataFrame) -> pd.DataFrame:
        return d.set_index("商品コード").rename(columns={
            "商品名": "name", "部門名": "dept", "純売上(税抜)": "sales", "原価": "cost",
            "販売点数": "units", "在庫数": "stock", "品番": "style", "カラー": "color", "サイズ": "size",
        })[["name", "style_name", "dept", "kind", "sales", "cost", "units", "stock", "style", "color", "size"]]

    a, b = per_sku(prev), per_sku(curr)
    sk = a.add_suffix("_25").join(b.add_suffix("_26"), how="outer").sort_index()
    for c in ["name", "style_name", "dept", "kind", "style", "color", "size"]:
        sk[c] = sk[c + "_26"].fillna(sk[c + "_25"])
    # The latest registration decides: a code re-registered as a fee or bag is not merchandise.
    sk = sk[sk["kind"] != "non_product"].copy()
    # 在庫数 is a snapshot taken at export time, so both files carry the same value.
    sk["stock"] = sk["stock_26"].fillna(sk["stock_25"])
    for c in ["sales", "cost", "units"]:
        for y in ["25", "26"]:
            sk[f"{c}_{y}"] = sk[f"{c}_{y}"].fillna(0)
    sk["gp_25"] = sk["sales_25"] - sk["cost_25"]
    sk["gp_26"] = sk["sales_26"] - sk["cost_26"]
    sk["stock_pos"] = sk["stock"].clip(lower=0)
    keep = [c for c in sk.columns if not c.endswith(("_25", "_26")) or c.split("_")[0] in ("sales", "cost", "units", "gp")]
    return sk[keep]


def resample(sk: pd.DataFrame, rng: np.random.Generator, window_days: int, shift_days: int) -> pd.DataFrame:
    """One stability draw: Poisson noise on unit sales, and the stock snapshot moved by up to ±shift_days.

    Money is scaled with units. A SKU that sold keeps at least one unit so the draw changes
    how well a product sold, not whether it existed.
    """
    sk = sk.copy()
    for y in ["25", "26"]:
        u = sk[f"units_{y}"].to_numpy()
        nu = rng.poisson(np.clip(u, 0, None)).astype(float)
        nu = np.where(u > 0, np.maximum(nu, 1), 0)
        f = np.where(u > 0, nu / np.where(u > 0, u, 1), 0)
        for c in ["sales", "cost", "gp"]:
            sk[f"{c}_{y}"] = sk[f"{c}_{y}"] * f
        sk[f"units_{y}"] = nu
    d = int(rng.integers(-shift_days, shift_days + 1))
    if d:
        flow = rng.poisson(sk["units_26"].to_numpy() / window_days * abs(d))
        sk["stock_pos"] = np.clip(sk["stock_pos"] - np.sign(d) * flow, 0, None)
        sk["stock"] = sk["stock_pos"]
    return sk


# --------------------------------------------------------------------------------------------
# Style metrics
# --------------------------------------------------------------------------------------------
def _style_frame(sk: pd.DataFrame, P: dict, cal: Calendar, frozen: pd.DataFrame | None) -> pd.DataFrame:
    sk = sk.copy()
    sk["dem"] = sk["units_25"] + sk["units_26"]
    sk["dem_in"] = np.where(sk["stock"] > 0, sk["dem"], 0)
    # The current best-selling SKU names the style and decides its shelf (products get renamed).
    top = sk.sort_values(["units_26", "units_25"], ascending=False).groupby("style")[["style_name", "name"]].first()
    # Keep every name's words for successor matching.
    name_words = sk.groupby("style")["style_name"].agg(lambda x: set().union(*map(tx.name_tokens, x)))
    compact = sk.groupby("style")["style_name"].agg(lambda x: {tx.compact_name(v) for v in x})
    st = sk.groupby("style").agg(
        dept=("dept", lambda x: x.mode().iloc[0]),
        s25=("sales_25", "sum"), g25=("gp_25", "sum"), u25=("units_25", "sum"),
        s26=("sales_26", "sum"), c26=("cost_26", "sum"), g26=("gp_26", "sum"), u26=("units_26", "sum"),
        stock=("stock_pos", "sum"), stock_raw=("stock", "sum"), n_sku=("name", "size"),
        dem=("dem", "sum"), dem_in=("dem_in", "sum"),
    )
    st = st.join(top)
    st["name_words"] = name_words.reindex(st.index)
    st["compact"] = compact.reindex(st.index)
    # Demand-weighted share of SKUs still in stock: low when the best-selling sizes/colours are gone.
    st["avail"] = np.where(st["dem"] > 0, st["dem_in"] / st["dem"].replace(0, np.nan), 0)
    st["pref"] = st.index.str[:3]
    st["seq"] = pd.to_numeric(st.index.str[3:], errors="coerce").fillna(0).astype(int)

    unknown = sorted(set(st["dept"]) - set(tx.DEPT_GROUP))
    if unknown:
        warnings.warn(f"棚グループが未設定の部門があります（小物として扱います）: {unknown}。taxonomy.DEPT_GROUP に追加してください。")
    st["grp"] = st["dept"].map(tx.DEPT_GROUP).fillna("小物")
    nm = st["name"].str.upper()
    is_outer = (nm.str.contains(tx.OUTER_KW) & ~nm.str.contains(tx.NOT_OUTER_KW)
                & st["grp"].isin(["スウェット・フーディ", "アウター"]))
    st.loc[is_outer, "grp"] = "アウター"
    st["lstee"] = ((st["grp"] == "スウェット・フーディ") & st["name"].str.contains(tx.LSTEE_KW, case=False)
                   & ~nm.str.contains(tx.NOT_LSTEE_KW))
    st.loc[st["lstee"], "grp"] = "Tシャツ"  # long-sleeve tees sit on the T-shirt shelf at the same price point
    st["heavy"] = (st["name"].str.contains(tx.HEAVY_KW, case=False) & ~nm.str.contains("UV")
                   & st["grp"].isin(TOPS + ["パンツ", "アウター"]))
    # Autumn/winter-skewed tops sell less inside a Jan–Oct window: correct their GP (tops only).
    st["sadj"] = np.where((st["heavy"] | st["lstee"]) & st["grp"].isin(TOPS), P["SEASON_HEAVY"], 1.0)
    st["winter"] = st["heavy"] | st["lstee"] | (st["grp"] == "アウター")

    if frozen is not None:
        st = st.join(frozen[["status", "age", "seqpct", "newpct"]])
    else:
        st["status"] = np.select([(st.u25 > 0) & (st.u26 > 0), st.u25 > 0, st.u26 > 0],
                                 ["both", "prev_only", "curr_only"], "none")
        # Code sequence as a launch-order proxy: percentile among codes that sold last year, per prefix.
        prev_codes = st[st.u25 > 0].groupby("pref")["seq"].apply(lambda x: np.sort(x.to_numpy())).to_dict()
        st["seqpct"] = [(prev_codes[p] <= q).mean() if p in prev_codes else 1.0 for p, q in zip(st["pref"], st["seq"])]
        # Among this year's new codes: how late in the year (by code order) the style was registered.
        new_codes = st[st.status == "curr_only"].groupby("pref")["seq"].apply(lambda x: np.sort(x.to_numpy())).to_dict()
        st["newpct"] = [(new_codes[p] <= q).mean() if len(new_codes.get(p, [])) >= P["NEWPCT_MIN_CODES"] else np.nan
                        for p, q in zip(st["pref"], st["seq"])]
        ratio = np.where(st.s25 > 0, st.s26 / st.s25.replace(0, np.nan), np.inf)
        recent = st["seqpct"] >= P["OLD_CODE_PCT"]
        st["age"] = "mature"
        leftover = ~recent & (st["u26"] <= P["LEFTOVER_MAX_UNITS"])
        st.loc[(st.status == "curr_only") & ~leftover, "age"] = "new"
        st.loc[(st.status == "both") & ((st.seqpct >= P["YOUNG_PCT"]) | (ratio >= P["LATE_RATIO"])), "age"] = "young"
        # A handful of units last year on one of last year's newest codes is a pre-sale or event,
        # not a season on sale: judge it as a new product.
        presale = ((st.status == "both") & (st.seqpct >= P["YOUNG_PCT"]) & (st.u25 <= P["PRESALE_MAX_UNITS"])
                   & (st.u26 >= np.maximum(P["NEW_MIN_UNITS"], 2 * st.u25)))
        st.loc[presale, "age"] = "new"
        # Launched late last year and sold out within it.
        st.loc[(st.status == "prev_only") & (st.seqpct >= P["YOUNG_PCT"]), "age"] = "young"

    st["series"] = st["name"].map(tx.series_of)
    st["collab"] = st["name"].str.contains(tx.COLLAB_KW)
    st["limited"] = st["name"].str.contains(tx.LIMITED_KW, case=False)

    # Money on the same footing: gross profit over this year's window, season-neutral.
    st["E_now"] = st["g26"] * st["sadj"]
    st["E_prev"] = st["g25"] * cal.w_gp * P["G_OWN"]
    # A style launched late last year only existed for part of it: do not pro-rate its GP down.
    young = st["age"] == "young"
    st.loc[young, "E_prev"] = np.maximum(st.loc[young, "g25"] * P["G_OWN"], st.loc[young, "E_prev"])
    st["gpu"] = (st.g26 / st.u26.replace(0, np.nan)).fillna(st.g25 / st.u25.replace(0, np.nan))
    st["gm26"] = st.g26 / st.s26.replace(0, np.nan)
    st["gm25"] = st.g25 / st.s25.replace(0, np.nan)
    # GP per unit at regular price: a marked-down year must not make the product look cheaper than it is.
    st["gpu_reg"] = np.fmax(st.g26 / st.u26.replace(0, np.nan), st.g25 / st.u25.replace(0, np.nan))
    p25 = st.s25 / st.u25.replace(0, np.nan)
    p26 = st.s26 / st.u26.replace(0, np.nan)
    st["price_ratio"] = p26 / p25
    return st


def _price_models(usable: pd.DataFrame, P: dict) -> dict:
    """ln(GP of launch) = a_class + b ln(GP per unit), per class (apparel / goods).

    A class with enough launches gets its own slope; otherwise it shares the pooled slope and
    keeps its own intercept. Slopes are clipped to [0, PRICE_ADJ_MAX_B].
    """
    x, y = np.log(usable.gpu.to_numpy()), np.log(usable.E_now.to_numpy())
    cls = usable["cls"].to_numpy()
    present = sorted(set(cls))
    X = np.column_stack([x] + [(cls == c).astype(float) for c in present])
    b_pool = float(np.clip(np.linalg.lstsq(X, y, rcond=None)[0][0], 0, P["PRICE_ADJ_MAX_B"]))
    a_pool = {c: float(np.mean(y[cls == c] - b_pool * x[cls == c])) for c in present}
    resid_pool = y - np.array([a_pool[c] for c in cls]) - b_pool * x
    models = {}
    for c in present:
        xc, yc = x[cls == c], y[cls == c]
        if len(xc) < P["MIN_CLASS_LAUNCHES"]:
            continue  # too few launches to place this class's intercept
        if len(xc) >= P["MIN_LAUNCH_FIT"] and len(np.unique(xc)) >= 3:
            b = float(np.clip(np.polyfit(xc, yc, 1)[0], 0, P["PRICE_ADJ_MAX_B"]))
            a = float(np.mean(yc - b * xc))
            resid = yc - a - b * xc
        else:
            b, a, resid = b_pool, a_pool[c], resid_pool
        models[c] = dict(a=a, b=b, n=len(xc), **{q: hd_quantile(resid, P[q]) for q in ["Q_LOW", "Q_LINE", "Q_HIGH"]})
    return models


def _hurdles(st: pd.DataFrame, P: dict, boot_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Break-point lines per shelf group from this year's genuine launches.

    Group quantiles (Harrell–Davis) are pooled in log space with a price-band model
    ln(GP of launch) = a + b ln(GP per unit), so groups with few launches borrow strength.
    """
    own = st[st["grp"] != tx.OTHER_GROUP]
    active = own[own.u26 > 0]
    gpu_g = active.groupby("grp")["gpu"].median()
    # The line is set by codes first sold this year (pre-sold styles had a head start).
    L = own[(own.age == "new") & (own.status == "curr_only") & (own.u26 >= P["NEW_MIN_UNITS"])
            & (own.seqpct >= P["OLD_CODE_PCT"])].copy()
    if boot_seed is not None:
        L = L.sample(len(L), replace=True, random_state=boot_seed)
    L["cls"] = np.where(L["grp"].isin(tx.GOODS_GROUPS), "goods", "apparel")
    usable = L[(L.E_now > 0) & (L.gpu > 0)]
    if len(usable) < 3 or usable["gpu"].nunique() < 3:
        raise ValueError("今期の新商品が少なすぎて分岐点ラインを計算できません（販売3点以上の新商品が3品番以上必要）。")
    model = _price_models(usable, P)
    rows = {}
    for g in gpu_g.index:
        Lg = L[L["grp"] == g]
        n = len(Lg)
        c = "goods" if g in tx.GOODS_GROUPS else "apparel"
        m = model.get(c)
        k = P["SHRINK_K"]
        # Price point the line refers to: the launches' and the shelf's typical GP per unit, blended like the line.
        gpu_ref = float(np.exp((n * np.log(Lg["gpu"].clip(lower=1).median()) if n else 0) / (n + k)
                               + k * np.log(gpu_g[g]) / (n + k)))
        row = {"n_launch": n, "gpu_med": gpu_g[g], "gpu_ref": gpu_ref, "b": m["b"] if m else 0.0, "basis": "新商品"}
        if m is None:
            # Too few launches in this class to model: anchor on the typical existing product of the shelf.
            warnings.warn(f"{g}: 今期の新商品が少ないため、既存品の分布を基準にラインを計算しています。")
            ex = active[(active.grp == g) & (active.E_now > 0)]["E_now"].to_numpy()
            if len(ex) == 0:
                ex = active[active.E_now > 0]["E_now"].to_numpy()
            row["basis"] = "既存品" if n == 0 else "新商品＋既存品"
        for q, lab in [("Q_LOW", "H25"), ("Q_LINE", "H50"), ("Q_HIGH", "H75")]:
            prior = hd_quantile(ex, P[q]) if m is None else float(np.exp(m["a"] + m["b"] * np.log(gpu_g[g]) + m[q]))
            raw = hd_quantile(Lg["E_now"].to_numpy(), P[q]) if n else prior
            row[f"{lab}_raw"] = raw if n else np.nan
            row[f"{lab}_prior"] = prior
            row[lab] = float(np.exp((n * np.log(max(raw, 1)) + k * np.log(prior)) / (n + k)))
        rows[g] = row
    return pd.DataFrame(rows).T, L, model


def _near(a: str, b: str) -> bool:
    """Edit distance at most 1 (a typo or a dropped letter)."""
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[i + 1:] == b[i + 1:] or a[i:] == b[i + 1:] or a[i + 1:] == b[i:]


def _same_name(a: str, b: str) -> bool:
    """Re-registered under a slightly different name, e.g. 'LAVENDER ME' -> 'NEW LUTEN LAVENDERME'."""
    if len(a) < 5 or len(b) < 5:
        return False
    if a in b or b in a:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) < 6:
        return False
    # Compare against every same-length window of the longer name, allowing one typo.
    return any(_near(short, long_[k:k + n]) for n in (len(short) - 1, len(short), len(short) + 1)
               for k in range(0, len(long_) - n + 1))


def _successors(st: pd.DataFrame, P: dict) -> pd.Series:
    """A later style on the same shelf that shares a distinctive name word (the shop already renewed it)."""
    later = st[(st.u26 > 0) & st.age.isin(["new", "young"]) & (st.grp != tx.OTHER_GROUP)]
    out = {}
    for i, r in st.iterrows():
        if r.grp == tx.OTHER_GROUP:
            out[i] = ""
            continue
        same_pref = later.pref == r.pref
        newer = (later.age == "new") | (later.seqpct - r.seqpct >= P["SUCC_GAP"])
        # Within a prefix the code must be later and not an adjacent sibling of the same batch.
        order_ok = ~same_pref | ((later.seq > r.seq + 2))
        c = later[(later.grp == r.grp) & (later.index != i) & newer & order_ok]
        hits = [f"{j}:{x.style_name}" for j, x in c.iterrows()
                if (r.name_words & x.name_words) or any(_same_name(a, b) for a in r.compact for b in x.compact)]
        out[i] = " / ".join(hits[:2])
    return pd.Series(out)


def compute_styles(sk: pd.DataFrame, P: dict, cal: Calendar, boot_seed: int | None = None,
                   frozen: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Metrics and labels for every style. Returns (styles, hurdles, launch sample).

    `frozen` carries the base run's life-stage columns into stability draws, so a draw only
    changes how well a product sold.
    """
    st = _style_frame(sk, P, cal, frozen)
    H, L, model = _hurdles(st, P, boot_seed)
    st = st.join(H[["H25", "H50", "H75", "b", "gpu_med", "gpu_ref"]], on="grp")
    if P["PRICE_ADJ"]:
        # A cheap tee's slot is compared with what a launch at the same price point earns.
        b = st["b"].astype(float).clip(0, P["PRICE_ADJ_MAX_B"])
        adj = (st["gpu_reg"].clip(lower=1) / st["gpu_ref"]) ** b
        adj = adj.clip(*P["PRICE_ADJ_CLIP"]).fillna(1.0)
    else:
        adj = pd.Series(1.0, index=st.index)
    st["price_adj"] = adj
    for c in ["H25", "H50", "H75"]:
        st[c] = st[c] * adj

    st["constrained"] = st["avail"] < P["AVAIL_MIN"]
    # Launch dates are unknown: a later code (by order among this year's new codes) sold for a shorter time.
    order = st["newpct"].astype(float).fillna(0.5)
    exposure = np.where(st["age"] == "new", 1 - order * (1 - P["NEW_EXPOSURE_MIN"]), 1.0)
    daily = st.u26 * st.sadj / (cal.window_days * exposure)
    rate_peak = daily * P["PEAK_MULT"]
    st["cover_peak"] = np.where(rate_peak > 0, st.stock / rate_peak.replace(0, np.nan), np.inf)
    st["cover_now"] = np.where(daily > 0, st.stock / daily.replace(0, np.nan), np.inf)
    st["DI"] = np.where(st.E_prev > 0, st.E_now / st.E_prev.replace(0, np.nan), np.nan)

    # Trend relative to in-stock mature styles on the same shelf (neutral to stock-outs and store growth).
    base = st[(st.grp != tx.OTHER_GROUP) & (st.age == "mature") & (st.status == "both") & ~st.constrained & (st.u25 >= 10)]
    all_med = (base.u26 * base.sadj / base.u25).median()
    grp_med = base.groupby("grp").apply(
        lambda d: (d.u26 * d.sadj / d.u25).median() if len(d) >= P["TI_MIN_GROUP"] else np.nan)
    st["norm"] = st["grp"].map(grp_med).fillna(all_med)
    st["TI"] = np.where(st.u25 > 0, st.u26 * st.sadj / st.u25.replace(0, np.nan) / st.norm, np.nan)
    cv = np.sqrt(1 / st.u25.replace(0, np.nan) + 1 / st.u26.replace(0, np.nan))
    st["TI_hi"] = st.TI * np.exp(P["TI_Z"] * cv)
    st["TI_lo"] = st.TI * np.exp(-P["TI_Z"] * cv)
    steady = (st.age == "mature") & (st.status == "both") & ~st.constrained
    st["declining"] = steady & (st.u25 >= P["TI_MIN_U25"]) & (st.TI < P["TI_DOWN"]) & (st.TI_hi < 1.0)
    st["growing"] = (st.status == "both") & ~st.constrained & (st.TI > P["TI_UP"]) & (st.TI_lo > 1.0)

    # Price/cost update: low margin AND low GP per unit, not a royalty-bound collaboration, capped price rise.
    active = st[(st.grp != tx.OTHER_GROUP) & (st.u26 > 0)]
    volume = active[active.u26 >= P["GM_MIN_UNITS"]]
    st["GM_low"] = st["grp"].map(volume.groupby("grp")["gm26"].quantile(P["GM_LOW_Q"]))
    st["GM50"] = st["grp"].map(volume.groupby("grp")["gm26"].median())
    need = ((1 - st.gm26) / (1 - st.GM50) - 1).clip(lower=0)
    st["price_up"] = np.minimum(need, P["PRICE_CAP"])
    st["uplift"] = st.s26 * st.sadj * st.price_up
    st["margin_case"] = ((st.gm26 < st.GM_low) & (st.gpu < st["gpu_med"]) & ~st.collab & (st.uplift >= P["MAT"]))

    dead = dead_skus(sk, st, P, cal)
    agg = dead.groupby("style").agg(dead_sku=("商品コード", "size"), dead_stock=("在庫", "sum"))
    st = st.join(agg)
    st[["dead_sku", "dead_stock"]] = st[["dead_sku", "dead_stock"]].fillna(0)
    st["overstock"] = (st["cover_now"] > P["OVERSTOCK_COVER"]) & (st.stock >= P["OVERSTOCK_MIN"])
    st["sku_prune"] = ((st.dead_sku >= 2)
                       | ((st.dead_stock > 0) & (st.dead_stock >= P["DEAD_STOCK_SHARE"] * st.stock.clip(lower=1)))
                       | st.overstock)

    st["missing_sizes"] = missing_sizes(sk, st, P)
    st["successor"] = _successors(st, P)
    # Implied sell-out date. A late launch's last-year units came from a few months, so they are
    # compared un-pro-rated (the date is then an upper bound).
    w_units = np.where(st["age"] == "young", 1.0, cal.w_units)
    q = np.where(st.u25 > 0, st.u26 * st.sadj / (st.u25.replace(0, np.nan) * w_units), np.nan)
    st["t_star"] = [implied_sellout_date(v, cal) for v in q]

    res = [_rule(r, P, cal.w_sales >= P["SEASON_FULL_YEAR_W"]) for r in st.itertuples()]
    st["rule"], st["label"], st["detail"] = zip(*res)
    keep_labels = ~st["label"].isin(STOP_LABELS + ("他社ブランド別管理",))
    has_gap = keep_labels & (st["missing_sizes"] != "")
    st.loc[has_gap, "detail"] = st.loc[has_gap, "detail"] + "／売れ筋SKUが今期ずっと欠品：" + st.loc[has_gap, "missing_sizes"] + " → 補充"

    lo, hi = P["BORDER"], 1 / P["BORDER"]
    ratio_now = st.E_now / st.H50
    ratio_prev = st.E_prev / (P["PROVEN_CHECK"] * st.H50)
    judged = (st.grp != tx.OTHER_GROUP) & st.age.isin(["mature", "young"])
    border_now = judged & (ratio_now >= lo) & (ratio_now < hi)
    # A sold-out style below the line is decided by last year's GP against 0.85×line.
    border_prev = judged & st.constrained & (st.E_now < st.H50) & (ratio_prev >= lo) & (ratio_prev < hi)
    st["border"] = border_now | border_prev
    st["line_ratio"] = ratio_now
    return st, H, L


def _sellout_text(t_star: str) -> str:
    if t_star == "期間末以降":
        return "今期も前年以上のペースで売れて欠品 → 同型で再生産"
    return f"{t_star} より前に欠品 → 同型で再生産／それ以降まで在庫あり → 刷新版"


def _prev_text(r) -> str:
    return "前年はライン超え" if r.E_prev >= r.H50 else f"前年はラインの{r.E_prev / r.H50:.0%}"


def _rule(r, P: dict, full_year: bool = False) -> tuple[str, str, str]:
    """Decision tree, evaluated top to bottom. Returns (rule id, label, reason).

    full_year: the window already contains the autumn/winter peak, so winter items are not deferred.
    """
    H25, H50, H75 = r.H25, r.H50, r.H75
    defer_winter = r.winter and not full_year
    if r.grp == tx.OTHER_GROUP:
        return "R0", "他社ブランド別管理", "ブランド単位で売上・掛率を管理（自社ラインでは判定しない）"
    if r.status == "prev_only" and r.stock > 0:
        if r.successor:
            return "R1", "終売（在庫消化）", f"今期販売ゼロの旧コード。後継（{r.successor}）と並べて定価で販売し、再生産しない"
        return "R1", "終売（在庫消化）", "今期販売ゼロの残在庫 → 値下げ・セットで処分、再生産なし"

    cleared = pd.notna(r.price_ratio) and r.price_ratio < P["CLEARANCE_PRICE"]
    if cleared and r.stock <= 0:
        return "R15", "終売済み", f"値下げ処分で完売（今期単価は前年の{r.price_ratio:.0%}）→ 再生産しない"
    short = r.constrained or r.cover_peak < P["COVER_REORDER"]
    est = "・発売日から推定" if r.age == "new" else ""
    why_short = ("欠品・売れ筋SKU切れ → 即手配" if r.constrained
                 else f"在庫{r.cover_peak:.0f}日分(ピーク換算{est}) → ピーク前に追加発注")
    pri = "A" if r.E_now >= H75 else "B"

    if r.age in ("new", "young"):
        tag = "新商品" if r.age == "new" else "前年後半投入"
        if r.E_now >= H50 and short:
            return "R2", "再生産", f"優先{pri}：{tag}、初速が新商品中央値超え、{why_short}"
        if r.E_now >= H50:
            return "R3", "継続・強化", f"{tag}で新商品中央値超え"
        if r.age == "new":
            if defer_winter:
                return "R4", "様子見", "冬物の新作：冬のピーク後に再判定（追加は小ロットで）"
            recent = pd.notna(r.newpct) and r.newpct >= P["NEW_RECENT_PCT"]
            if recent and not full_year:
                return "R4", "様子見", "発売から日が浅い可能性が高い新作：次回の判定で再評価"
            if r.E_now < H25 and r.stock <= 0:
                return "R4", "様子見", "新商品・少量で完売（新商品下位25%未満）→ 再生産するなら需要を確かめて小ロットで"
            if r.E_now < H25 and r.stock > 0:
                return "R4", "様子見", "新商品・弱いスタート（新商品下位25%未満）→ 追加生産しない・次回の判定で再評価"
            if r.constrained and r.E_now >= H25:
                return "R4", "様子見", "新商品・完売（ライン手前）→ 追加は小ロットで・次回の判定で再評価"
            return "R4", "様子見", "新商品・次回の判定で再評価"
        if r.stock > 0:
            if defer_winter:
                return "R4", "様子見", "前年後半投入の秋冬物：冬のピーク後に再判定（今冬の追加生産は見送り）"
            if r.E_now < H25:
                return "R4b", "終売（在庫消化）", "前年後半投入・今期を通して新商品下位25%未満 → 定価で売り切り・次回生産なし"
            return "R4", "様子見", "前年後半投入・ライン手前 → 次回の判定で再評価"
        # young, sold out and below the line: judged like a mature style below

    if r.age == "mature" and r.E_now >= H50:
        if short:
            if r.constrained and (r.DI if pd.notna(r.DI) else 1) < P["DI_CHECK"]:
                return "R5c", "再生産・復刻（要確認）", f"欠品中・前年比{r.DI:.2f}：{_sellout_text(r.t_star)}"
            return "R5", "再生産", f"優先{pri}：ライン超え、{why_short}"
        if r.margin_case:
            return ("R6", "アップデート",
                    f"価格・原価（粗利率{r.gm26:.0%}が棚の下位20%、+{r.price_up:.0%}値上げで粗利+{r.uplift / 1e3:.0f}千円）")
        if r.declining:
            note = "。長袖は秋冬の売れ方の影響を受けやすいので冬明けに再確認" if r.lstee else ""
            return "R7", "アップデート", f"デザイン刷新（在庫があるのに同じ棚の中で有意に減速、TI={r.TI:.2f}{note}）"
        if r.sku_prune:
            what = (f"死に筋SKU{int(r.dead_sku)}・滞留{int(r.dead_stock)}点を次回生産から外す" if r.dead_sku > 0
                    else "在庫1年分超 → 色・サイズを絞る")
            return "R8", "継続・強化", f"ライン超え＋SKU整理：{what}"
        return "R9", "継続・強化", "ライン超え"

    # ---- below the break-point line
    proven_prev = r.E_prev >= P["PROVEN_CHECK"] * H50
    if r.constrained and proven_prev and not cleared:
        if r.E_prev < P["MAT_CHECK"]:
            return ("R15", "終売済み" if r.stock <= 0 else "終売（在庫消化）",
                    "前年はライン超えだが少額品のため確認対象外 → 再生産は任意")
        if r.limited:
            return ("R10", "終売（在庫消化）" if r.stock > 0 else "終売済み", "限定品：前年はライン超えだが再生産しない")
        if r.successor and r.stock <= 0:
            return "R10", "終売済み", f"後継あり（{r.successor}）→ 後継の数字で判断"
        if r.status == "prev_only":
            return "R10", "再生産・復刻（要確認）", f"前年中に完売・今期は未投入（{_prev_text(r)}）：意図的な終売でなければ復刻（同型 or 刷新版）"
        if pd.notna(r.gm25) and pd.notna(r.gm26) and r.gm26 < r.gm25 - P["DISCOUNT_GM_DROP"]:
            return "R10", "再生産・復刻（要確認）", f"欠品中・{_prev_text(r)}：今期は値引き販売で完売 → 再生産するかは前年の数字で判断"
        return "R10", "再生産・復刻（要確認）", f"欠品中・{_prev_text(r)}：{_sellout_text(r.t_star)}"
    if not r.constrained and r.growing and r.E_now >= P["GROW_FLOOR"] * H50:
        return "R11", "様子見", f"成長中（同じ棚の中でTI={r.TI:.2f}）→ 来季に再判定"
    if not r.constrained and r.E_prev >= max(H50, P["MAT_CHECK"]):
        note = "。長袖は秋冬の売れ方の影響を受けやすいので冬明けに再確認" if r.lstee else ""
        return "R12", "アップデート", f"デザイン刷新（在庫があったのに、前年ライン超え → 今期ライン割れ{note}）"
    if not r.constrained and r.margin_case and r.E_now + r.uplift >= H50:
        return "R13", "アップデート", f"価格・原価（+{r.price_up:.0%}の値上げでラインに届く）"
    if r.stock > 0:
        if r.successor:
            sub = f"後継（{r.successor}）と並べて定価で販売・次回生産なし"
        elif defer_winter:
            sub = "秋冬物：今冬の追加生産なし・冬は定価販売・冬明けに値下げ"
        elif r.E_now < H25 and not r.growing and (r.cover_now > P["MARKDOWN_COVER"] or r.declining):
            sub = f"即時：値下げ・セットで消化（今のペースで在庫{min(r.cover_now, 9999):.0f}日分）、再生産なし"
        else:
            sub = "定価で売り切り・次回生産なし"
        return "R14", "終売（在庫消化）", sub
    return "R15", "終売済み", "在庫0・2年ともライン未達 → 再生産しない"


# --------------------------------------------------------------------------------------------
# Secondary outputs
# --------------------------------------------------------------------------------------------
def missing_sizes(sk: pd.DataFrame, st: pd.DataFrame, P: dict) -> pd.Series:
    """Colour/size SKUs that carried a big share of last year's sales but sold nothing and had no stock this year.

    A same colour/size SKU registered under a new code that did sell is not counted as missing.
    """
    s = sk[sk["style"].isin(st[st.status == "both"].index)].copy()
    s["colour"] = s["color"].map(normalize_color)
    s["share25"] = s["units_25"] / s.groupby("style")["units_25"].transform("sum")
    now = s[s.units_26 > 0]
    sold = set(zip(now["style"], now["colour"], now["size"]))
    sold_size = set(zip(now["style"], now["size"]))
    gap = s[(s.share25 >= P["MISSING_SHARE"]) & (s.units_25 >= P["MISSING_MIN_UNITS"])
            & (s.units_26 == 0) & (s.stock_pos == 0)]
    # A colour-less placeholder code ('-') is covered by any colour of the same size.
    covered = [((a, c) in sold_size) if b == "-" else ((a, b, c) in sold)
               for a, b, c in zip(gap["style"], gap["colour"], gap["size"])]
    gap = gap[~np.array(covered, dtype=bool)]
    text = gap.groupby("style").apply(lambda d: "・".join(f"{c} {z}" for c, z in zip(d["color"], d["size"])))
    return text.reindex(st.index).fillna("")


def dead_skus(sk: pd.DataFrame, st: pd.DataFrame, P: dict, cal: Calendar) -> pd.DataFrame:
    """Colour/size SKUs of active own styles whose stock sits while the rest of the style sells."""
    s = sk[sk["style"].isin(st[(st.grp != tx.OTHER_GROUP) & (st.u26 > 0)].index)].copy()
    g = s.groupby("style")
    s["su"] = g["units_26"].transform("sum")
    s["ns"] = g["units_26"].transform("size")
    s["sku_cover"] = np.where(s.units_26 > 0, s.stock_pos / (s.units_26 / cal.window_days).replace(0, np.nan), np.inf)
    dead = ((s.ns >= P["DEAD_MIN_SKU"]) & (s.su >= P["DEAD_MIN_UNITS"]) & (s.units_26 / s.su < P["DEAD_SHARE"] / s.ns)
            & (s.stock_pos >= P["DEAD_MIN_STOCK"]) & (s.sku_cover > P["DEAD_COVER"]))
    out = s[dead].reset_index()[["商品コード", "style", "name", "color", "size", "units_26", "stock_pos", "sku_cover"]]
    return out.rename(columns={"units_26": "今期販売点数", "stock_pos": "在庫", "sku_cover": "在庫日数"})


def series_table(st: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Health of each design series: where a freed slot should go (same series or a new concept)."""
    own = st[(st.grp != tx.OTHER_GROUP) & (st.series != tx.NO_SERIES)]
    rows = []
    for f, d in own.groupby("series"):
        m = d[(d.age == "mature") & (d.status == "both") & ~d.constrained]
        denom = (m.u25 * m.norm).sum()
        fi = (m.u26 * m.sadj).sum() / denom if len(m) and denom > 0 else np.nan
        nw = d[d.age.isin(["new", "young"])]
        rows.append(dict(series=f, n_active=int((d.u26 > 0).sum()), n_prev_only=int((d.status == "prev_only").sum()),
                         s25=d.s25.sum(), s26=d.s26.sum(), FI_instock=fi, n_new_or_young=len(nw),
                         new_hits=int((nw.E_now >= nw.H50).sum())))
    ft = pd.DataFrame(rows).set_index("series")
    healthy = ft.FI_instock >= P["SERIES_FI_OK"]
    hits = ft.new_hits >= 1
    tried = ft.n_new_or_young >= 1
    ft["direction"] = np.select(
        [healthy & hits, healthy | hits, ft.FI_instock.notna() | tried],
        ["同シリーズで新柄・新型", "同シリーズも可（新作の当たりを確認）", "新コンセプトへ"], "判定不能")
    return ft.sort_values("s26", ascending=False)


def brand_table(st: pd.DataFrame, cal: Calendar) -> pd.DataFrame:
    """Other-brand (ETC1) codes summed per brand, with same-window growth."""
    e = st[st.grp == tx.OTHER_GROUP].copy()
    e["brand"] = e["name"].str.split().str[0]
    b = e.groupby("brand").agg(codes=("name", "size"), s25=("s25", "sum"), s26=("s26", "sum"),
                               u25=("u25", "sum"), u26=("u26", "sum"))
    b["same_window_ratio"] = b.s26 / (b.s25 * cal.w_sales).replace(0, np.nan)
    b["share26"] = b.s26 / st.s26.sum()
    return b.sort_values("s26", ascending=False)


def stability(sk: pd.DataFrame, P: dict, cal: Calendar, base: pd.DataFrame, runs: int,
              seed: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run the logic on resampled sales, a moved stock snapshot and a bootstrapped launch sample.

    stab = share of runs that give the same label as the base run; stop_freq = share labelled 終売.
    """
    frozen = base[["status", "age", "seqpct", "newpct"]]
    labels, lines = {}, []
    for i in range(runs):
        rng = np.random.default_rng(seed + i)
        draw = resample(sk, rng, cal.window_days, P["STAB_SHIFT_DAYS"])
        try:
            st_i, H_i, _ = compute_styles(draw, P, cal, boot_seed=seed + 1000 + i, frozen=frozen)
        except ValueError:
            # Too few distinct launches in this bootstrap draw: keep the sales noise, skip the launch resampling.
            st_i, H_i, _ = compute_styles(draw, P, cal, boot_seed=None, frozen=frozen)
        labels[i] = st_i["label"].reindex(base.index)
        lines.append(H_i["H50"])
    Lb = pd.DataFrame(labels)
    out = pd.DataFrame({"stab": Lb.eq(base["label"], axis=0).mean(axis=1),
                        "stop_freq": Lb.isin(STOP_LABELS).mean(axis=1)})
    return out, pd.DataFrame(lines).quantile([0.1, 0.5, 0.9]).T
