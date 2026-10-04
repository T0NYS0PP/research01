"""Stop-selling break-point logic per style (品番).

Break-point: a style should stop being sold once the gross profit it is expected to earn
falls below what a typical NEW product earns on the same shelf (median first-period gross
profit of this year's launches, H50). Stock-outs, launch timing and winter seasonality are
corrected before the comparison, and every result carries the reason and a stability score.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from . import taxonomy as tx
from .params import Calendar, implied_sellout_date

STOP_LABELS = ("終売（在庫消化）", "終売済み")
LABEL_ORDER = ["再生産", "再生産・復刻（要確認）", "継続・強化", "アップデート", "様子見",
               "終売（在庫消化）", "終売済み", "他社ブランド別管理"]

_GRID = np.linspace(0, 1, 4001)


def hd_quantile(x, p: float) -> float:
    """Harrell–Davis quantile: a weighted mean of all order statistics, steadier than the sample quantile."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    a, b = (n + 1) * p, (n + 1) * (1 - p)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    g = np.clip(_GRID, 1e-300, 1)
    with np.errstate(divide="ignore"):
        pdf = np.exp((a - 1) * np.log(g) + (b - 1) * np.log(np.clip(1 - _GRID, 1e-300, 1)) - log_beta)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(_GRID))])
    cdf /= cdf[-1]
    w = np.diff(np.interp(np.arange(n + 1) / n, _GRID, cdf))
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


def resample_units(sk: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Poisson-resample unit sales per SKU (money scaled in proportion) for the stability test."""
    sk = sk.copy()
    for y in ["25", "26"]:
        u = sk[f"units_{y}"].to_numpy()
        nu = rng.poisson(np.clip(u, 0, None)).astype(float)
        f = np.where(u > 0, nu / np.where(u > 0, u, 1), 0)
        for c in ["sales", "cost", "gp"]:
            sk[f"{c}_{y}"] = sk[f"{c}_{y}"] * f
        sk[f"units_{y}"] = nu
    return sk


# --------------------------------------------------------------------------------------------
# Style metrics
# --------------------------------------------------------------------------------------------
def _style_frame(sk: pd.DataFrame, P: dict, cal: Calendar) -> pd.DataFrame:
    sk = sk.copy()
    sk["dem"] = sk["units_25"] + sk["units_26"]
    sk["dem_in"] = np.where(sk["stock"] > 0, sk["dem"], 0)
    # Display name: the best-selling SKU of the current year (falls back to last year).
    top_sku = sk.sort_values(["units_26", "units_25"], ascending=False).groupby("style")["style_name"].first()
    # Products are sometimes renamed between years; keep every name's words for successor matching.
    name_words = sk.groupby("style")["style_name"].agg(lambda x: set().union(*map(tx.name_tokens, x)))
    st = sk.groupby("style").agg(
        name=("name", "first"), dept=("dept", lambda x: x.mode().iloc[0]),
        s25=("sales_25", "sum"), g25=("gp_25", "sum"), u25=("units_25", "sum"),
        s26=("sales_26", "sum"), c26=("cost_26", "sum"), g26=("gp_26", "sum"), u26=("units_26", "sum"),
        stock=("stock_pos", "sum"), stock_raw=("stock", "sum"), n_sku=("name", "size"),
        dem=("dem", "sum"), dem_in=("dem_in", "sum"),
    )
    st["style_name"] = top_sku.reindex(st.index)
    st["name_words"] = name_words.reindex(st.index)
    # Demand-weighted share of SKUs still in stock: low when the best-selling sizes/colours are gone.
    st["avail"] = np.where(st["dem"] > 0, st["dem_in"] / st["dem"].replace(0, np.nan), 0)
    st["pref"] = st.index.str[:3]
    st["seq"] = pd.to_numeric(st.index.str[3:], errors="coerce").fillna(0).astype(int)
    st["status"] = np.select([(st.u25 > 0) & (st.u26 > 0), st.u25 > 0], ["both", "prev_only"], "curr_only")

    st["grp"] = st["dept"].map(tx.DEPT_GROUP).fillna("小物")
    nm = st["name"].str.upper()
    is_outer = (nm.str.contains(tx.OUTER_KW) & ~nm.str.contains(tx.NOT_OUTER_KW)
                & st["grp"].isin(["スウェット・フーディ", "アウター"]))
    st.loc[is_outer, "grp"] = "アウター"
    is_lstee = ((st["grp"] == "スウェット・フーディ") & st["name"].str.contains(tx.LSTEE_KW, case=False)
                & ~nm.str.contains(tx.NOT_LSTEE_KW))
    st.loc[is_lstee, "grp"] = "Tシャツ"  # long-sleeve tees sit on the T-shirt shelf at the same price point
    st["heavy"] = (st["name"].str.contains(tx.HEAVY_KW, case=False) & ~nm.str.contains("UV")
                   & st["grp"].isin(["Tシャツ", "スウェット・フーディ", "パンツ", "キッズ", "アウター"]))
    st["sadj"] = np.where(st["heavy"] & (st["grp"] != "アウター"), P["SEASON_HEAVY"], 1.0)
    st["winter"] = st["heavy"] | (st["grp"] == "アウター")

    # Code sequence as a launch-order proxy: percentile among codes that sold last year, per prefix.
    prev_codes = st[st.u25 > 0].groupby("pref")["seq"].apply(lambda x: np.sort(x.to_numpy())).to_dict()
    st["seqpct"] = [(prev_codes[p] <= q).mean() if p in prev_codes else 1.0 for p, q in zip(st["pref"], st["seq"])]
    st["ratio"] = np.where(st.s25 > 0, st.s26 / st.s25.replace(0, np.nan), np.inf)
    st["age"] = "mature"
    leftover = (st["seqpct"] < P["OLD_CODE_PCT"]) & (st["u26"] <= P["LEFTOVER_MAX_UNITS"])
    st.loc[(st.status == "curr_only") & ~leftover, "age"] = "new"
    st.loc[(st.status == "both") & ((st.seqpct >= P["YOUNG_PCT"]) | (st.ratio >= P["LATE_RATIO"])), "age"] = "young"

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
    return st


def _hurdles(st: pd.DataFrame, P: dict, boot_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Break-point lines per shelf group from this year's genuine launches.

    Group quantiles (Harrell–Davis) are pooled in log space with a price-band model
    ln(GP of launch) = a + b ln(GP per unit), so groups with few launches borrow strength.
    """
    own = st[st["grp"] != tx.OTHER_GROUP]
    active = own[own.u26 > 0]
    gpu_g = active.groupby("grp")["gpu"].median()
    L = own[(own.age == "new") & (own.u26 >= P["NEW_MIN_UNITS"]) & (own.seqpct >= P["OLD_CODE_PCT"])].copy()
    if boot_seed is not None:
        L = L.sample(len(L), replace=True, random_state=boot_seed)
    L["cls"] = np.where(L["grp"].isin(tx.GOODS_GROUPS), "goods", "apparel")
    model = {}
    for c in ["goods", "apparel"]:
        d = L[(L.cls == c) & (L.E_now > 0) & (L.gpu > 0)]
        b, a = np.polyfit(np.log(d.gpu), np.log(d.E_now), 1)
        resid = np.log(d.E_now) - (a + b * np.log(d.gpu))
        model[c] = dict(a=a, b=b, n=len(d), **{q: hd_quantile(resid, P[q]) for q in ["Q_LOW", "Q_LINE", "Q_HIGH"]})
    rows = {}
    for g in gpu_g.index:
        Lg = L[L["grp"] == g]
        n = len(Lg)
        m = model["goods" if g in tx.GOODS_GROUPS else "apparel"]
        row = {"n_launch": n, "gpu_med": gpu_g[g]}
        for q, lab in [("Q_LOW", "H25"), ("Q_LINE", "H50"), ("Q_HIGH", "H75")]:
            prior = float(np.exp(m["a"] + m["b"] * np.log(gpu_g[g]) + m[q]))
            raw = hd_quantile(Lg["E_now"].to_numpy(), P[q]) if n else prior
            row[f"{lab}_raw"] = raw if n else np.nan
            row[f"{lab}_prior"] = prior
            row[lab] = float(np.exp((n * np.log(max(raw, 1)) + P["SHRINK_K"] * np.log(prior)) / (n + P["SHRINK_K"])))
        rows[g] = row
    return pd.DataFrame(rows).T, L


def _successors(st: pd.DataFrame, P: dict) -> pd.Series:
    """A later style on the same shelf that shares a distinctive name word (the shop already renewed it)."""
    toks = st["name_words"]
    later = st[(st.u26 > 0) & st.age.isin(["new", "young"]) & (st.grp != tx.OTHER_GROUP)]
    out = {}
    for i, r in st.iterrows():
        if r.grp == tx.OTHER_GROUP:
            out[i] = ""
            continue
        c = later[(later.grp == r.grp) & (later.seq > r.seq) & (later.index != i)
                  & ((later.age == "new") | (later.seqpct - r.seqpct >= P["SUCC_GAP"]))]
        hits = [f"{j}:{x.style_name}" for j, x in c.iterrows() if toks[i] & toks[j]]
        out[i] = " / ".join(hits[:2])
    return pd.Series(out)


def compute_styles(sk: pd.DataFrame, P: dict, cal: Calendar,
                   boot_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Metrics and labels for every style. Returns (styles, hurdles, launch sample)."""
    st = _style_frame(sk, P, cal)
    H, L = _hurdles(st, P, boot_seed)
    st = st.join(H[["H25", "H50", "H75"]], on="grp")

    st["constrained"] = st["avail"] < P["AVAIL_MIN"]
    rate_peak = st.u26 * st.sadj / cal.window_days * P["PEAK_MULT"]
    st["cover_peak"] = np.where(rate_peak > 0, st.stock / rate_peak.replace(0, np.nan), np.inf)
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
    st["growing"] = steady & (st.TI > P["TI_UP"]) & (st.TI_lo > 1.0)

    # Price/cost update: low margin AND low GP per unit, not a royalty-bound collaboration, capped price rise.
    active = st[(st.grp != tx.OTHER_GROUP) & (st.u26 > 0)]
    st["GM_low"] = st["grp"].map(active.groupby("grp")["gm26"].quantile(P["GM_LOW_Q"]))
    st["GM50"] = st["grp"].map(active.groupby("grp")["gm26"].median())
    gpu_g = active.groupby("grp")["gpu"].median()
    need = ((1 - st.gm26) / (1 - st.GM50) - 1).clip(lower=0)
    st["price_up"] = np.minimum(need, P["PRICE_CAP"])
    st["uplift"] = st.s26 * st.sadj * st.price_up
    st["margin_case"] = ((st.gm26 < st.GM_low) & (st.gpu < st["grp"].map(gpu_g)) & ~st.collab
                         & (st.uplift >= P["MAT"]))

    dead = dead_skus(sk, st, P, cal)
    agg = dead.groupby("style").agg(dead_sku=("商品コード", "size"), dead_stock=("在庫", "sum"))
    st = st.join(agg)
    st[["dead_sku", "dead_stock"]] = st[["dead_sku", "dead_stock"]].fillna(0)
    cover_all = np.where(st.u26 > 0, st.stock / (st.u26 / cal.window_days).replace(0, np.nan), np.inf)
    st["overstock"] = (cover_all > P["OVERSTOCK_COVER"]) & (st.stock >= P["OVERSTOCK_MIN"])
    st["sku_prune"] = ((st.dead_sku >= 2)
                       | ((st.dead_stock > 0) & (st.dead_stock >= P["DEAD_STOCK_SHARE"] * st.stock.clip(lower=1)))
                       | st.overstock)

    st["successor"] = _successors(st, P)
    q = np.where(st.u25 > 0, st.u26 * st.sadj / (st.u25.replace(0, np.nan) * cal.w_units), np.nan)
    st["t_star"] = [implied_sellout_date(v, cal) for v in q]

    res = [_rule(r, P) for r in st.itertuples()]
    st["rule"], st["label"], st["detail"] = zip(*res)

    ref = np.where(st.label.isin(STOP_LABELS), np.maximum(st.E_now, np.where(st.constrained, st.E_prev, 0)), st.E_now)
    st["border"] = ((st.grp != tx.OTHER_GROUP) & (ref >= P["BORDER"] * st.H50) & (ref < st.H50 / P["BORDER"])
                    & st.age.isin(["mature", "young"]))
    st["line_ratio"] = st.E_now / st.H50
    return st, H, L


def _rule(r, P: dict) -> tuple[str, str, str]:
    """Decision tree, evaluated top to bottom. Returns (rule id, label, reason)."""
    H25, H50, H75 = r.H25, r.H50, r.H75
    if r.grp == tx.OTHER_GROUP:
        return "R0", "他社ブランド別管理", "ブランド単位で売上・掛率を管理（自社ラインでは判定しない）"
    if r.status == "prev_only" and r.stock > 0:
        return "R1", "終売（在庫消化）", "今期販売ゼロの残在庫 → 値下げ・セットで処分、再生産なし"

    short = r.constrained or r.cover_peak < P["COVER_REORDER"]
    why_short = "欠品・売れ筋SKU切れ" if r.constrained else f"在庫{r.cover_peak:.0f}日分(ピーク換算)"
    pri = "A" if r.E_now >= H75 else "B"

    if r.age in ("new", "young"):
        tag = "新商品" if r.age == "new" else "前年後半投入"
        if r.E_now >= H50 and short:
            return "R2", "再生産", f"優先{pri}・即手配（{tag}、初速が新商品中央値超え、{why_short}）"
        if r.E_now >= H50:
            return "R3", "継続・強化", f"{tag}で新商品中央値超え"
        if r.age == "new":
            if r.E_now < H25 and r.stock > 0:
                return "R4", "様子見", "新商品・弱いスタート（新商品下位25%未満）→ 追加生産しない・通年で再判定"
            if r.constrained and r.E_now >= H25:
                return "R4", "様子見", "新商品・完売（ライン手前）→ 追加は小ロットで・通年で再判定"
            return "R4", "様子見", "新商品・通年で再判定"
        if r.stock > 0:
            if r.winter:
                return "R4", "様子見", "前年後半投入の冬物・冬明けに再判定（今冬の追加生産は見送り）"
            if r.E_now < H25:
                return "R4b", "終売（在庫消化）", "前年後半投入・1シーズン経過で新商品下位25%未満 → 定価で売り切り・次回生産なし"
            return "R4", "様子見", "前年後半投入・ライン手前 → 冬明けに再判定"
        # young, sold out and below the line: judged like a mature style below

    if r.age == "mature" and r.E_now >= H50:
        if short:
            if r.constrained and (r.DI if pd.notna(r.DI) else 1) < P["DI_CHECK"]:
                return ("R5c", "再生産・復刻（要確認）",
                        f"欠品中・前年比{r.DI:.2f}：{r.t_star} より前に欠品 → 同型で再生産／それ以降まで在庫あり → 刷新版")
            return "R5", "再生産", f"優先{pri}・即手配（ライン超え、{why_short}）"
        if r.margin_case:
            return ("R6", "アップデート",
                    f"価格・原価（粗利率{r.gm26:.0%}が棚の下位20%、+{r.price_up:.0%}値上げで粗利+{r.uplift / 1e3:.0f}千円）")
        if r.declining:
            return "R7", "アップデート", f"デザイン刷新（在庫があるのに同じ棚の中で有意に減速、TI={r.TI:.2f}）"
        if r.sku_prune:
            what = (f"死に筋SKU{int(r.dead_sku)}・滞留{int(r.dead_stock)}点を次回生産から外す" if r.dead_sku > 0
                    else "在庫1年分超 → 色・サイズを絞る")
            return "R8", "継続・強化", f"ライン超え＋SKU整理：{what}"
        return "R9", "継続・強化", "ライン超え"

    # ---- below the break-point line
    proven = r.E_prev >= max(H50, P["MAT_CHECK"])
    if r.constrained and r.E_prev >= max(P["PROVEN_CHECK"] * H50, P["MAT_CHECK"]):
        if r.limited:
            return ("R10", "終売（在庫消化）" if r.stock > 0 else "終売済み", "限定品：前年はライン超えだが再生産しない")
        if r.successor and r.stock <= 0:
            return "R10", "終売済み", f"後継あり（{r.successor}）→ 後継の数字で判断"
        if r.status == "prev_only":
            return "R10", "再生産・復刻（要確認）", "前年中に完売・今期は未投入：意図的な終売でなければ復刻（同型 or 刷新版）"
        return ("R10", "再生産・復刻（要確認）",
                f"欠品中・前年はライン超え：{r.t_star} より前に欠品 → 同型で再生産／それ以降まで在庫あり → 刷新版")
    if not r.constrained and r.growing and r.E_now >= P["GROW_FLOOR"] * H50:
        return "R11", "様子見", f"成長中（同じ棚の中でTI={r.TI:.2f}）→ 来季に再判定"
    if not r.constrained and proven:
        return "R12", "アップデート", "デザイン刷新（在庫があったのに、前年ライン超え → 今期ライン割れ）"
    if not r.constrained and r.margin_case and r.E_now + r.uplift >= H50:
        return "R13", "アップデート", f"価格・原価（+{r.price_up:.0%}の値上げでラインに届く）"
    if r.stock > 0:
        if r.winter:
            sub = "冬物：今冬の追加生産なし・冬は定価販売・冬明けに値下げ"
        elif r.E_now < H25:
            sub = "即時：値下げ・セットで消化、再生産なし"
        else:
            sub = "定価で売り切り・次回生産なし"
        return "R14", "終売（在庫消化）", sub
    return "R15", "終売済み", "在庫0・2年ともライン未達 → 再生産しない"


# --------------------------------------------------------------------------------------------
# Secondary outputs
# --------------------------------------------------------------------------------------------
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


def series_table(st: pd.DataFrame) -> pd.DataFrame:
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
    ft["direction"] = np.where((ft.FI_instock >= 0.8) | (ft.new_hits >= 1), "同シリーズで新柄・新型",
                               np.where(ft.FI_instock.isna(), "判定不能", "新コンセプトへ"))
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
    """Re-run the logic on Poisson-resampled sales and a bootstrapped launch sample.

    stab = share of runs that give the same label as the base run; stop_freq = share labelled 終売.
    """
    labels, lines = {}, []
    for i in range(runs):
        rng = np.random.default_rng(seed + i)
        st_i, H_i, _ = compute_styles(resample_units(sk, rng), P, cal, boot_seed=seed + 1000 + i)
        labels[i] = st_i["label"].reindex(base.index)
        lines.append(H_i["H50"])
    Lb = pd.DataFrame(labels)
    out = pd.DataFrame({"stab": Lb.eq(base["label"], axis=0).mean(axis=1),
                        "stop_freq": Lb.isin(STOP_LABELS).mean(axis=1)})
    return out, pd.DataFrame(lines).quantile([0.1, 0.5, 0.9]).T
