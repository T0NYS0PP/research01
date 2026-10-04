"""Adjustable parameters of the stop-selling logic, and calendar factors derived from monthly sales.

Every threshold the decision rules use lives in DEFAULT_PARAMS so the shop can tune it
without touching the rules. Defaults were calibrated on the 2025 / 2026-YTD exports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict = dict(
    # --- seasonality / growth -------------------------------------------------------
    SEASON_HEAVY=1.25,      # 重衣料トップス(フーディ・ニット等)の今期粗利に掛ける季節補正
    G_OWN=None,             # 自社品の同期間粗利成長率。None なら月次データから自動計算
    # --- life stage -----------------------------------------------------------------
    NEW_MIN_UNITS=3,        # 新商品ラインの標本に入れる最低販売点数
    OLD_CODE_PCT=0.50,      # 今期初販売でも前年コードの中央値より古い品番は「残り在庫」
    LEFTOVER_MAX_UNITS=2,   # ↑のうち販売点数がこれ以下なら成熟品として扱う
    YOUNG_PCT=0.80,         # 両年販売で、連番が前年販売コードの上位20%なら「前年後半投入」
    LATE_RATIO=2.75,        # 今期/前年の売上比がこれ以上でも「前年後半投入」
    # --- break-point line (新商品ライン) ----------------------------------------------
    Q_LINE=0.50,            # 分岐点ライン＝新商品の初期粗利の中央値
    Q_LOW=0.25,             # 即時値下げライン
    Q_HIGH=0.75,            # 再生産の優先Aライン
    SHRINK_K=5,             # 新商品が少ない棚を価格帯モデルに寄せる擬似件数
    # --- supply / stock -------------------------------------------------------------
    AVAIL_MIN=0.50,         # 有効在庫率がこれ未満なら「欠品（供給制約）」
    COVER_REORDER=21,       # ピーク換算の在庫日数がこれ未満なら再生産を急ぐ
    PEAK_MULT=None,         # ピーク月の日販 ÷ 期間平均日販。None なら月次から自動計算
    DI_CHECK=0.50,          # 欠品中で前年比がこれ未満なら「再生産・復刻（要確認）」
    PROVEN_CHECK=0.85,      # 欠品品は前年粗利がラインの85%以上で「需要実証済み」
    MAT_CHECK=50_000,       # ↑の最低粗利（パッチ等の少額品を確認対象から外す）
    # --- trend (同じ棚の在庫あり品と比べた相対トレンド) ----------------------------------
    TI_DOWN=0.60, TI_UP=1.20, TI_Z=1.28, TI_MIN_U25=30, TI_MIN_GROUP=8,
    GROW_FLOOR=0.70,        # 成長中の例外はラインの70%以上の品番だけ
    # --- price / cost update --------------------------------------------------------
    GM_LOW_Q=0.20,          # 粗利率が棚の下位20%なら価格・原価の見直し候補
    PRICE_CAP=0.15,         # 想定する値上げ幅の上限
    MAT=50_000,             # 見直しで増える粗利の最低額
    # --- SKU pruning ----------------------------------------------------------------
    DEAD_SHARE=0.25, DEAD_MIN_STOCK=2, DEAD_COVER=180, DEAD_MIN_SKU=3, DEAD_MIN_UNITS=30,
    DEAD_STOCK_SHARE=0.20, OVERSTOCK_COVER=365, OVERSTOCK_MIN=10,
    # --- misc -----------------------------------------------------------------------
    SUCC_GAP=0.20,          # 後継品番は今期新商品か、連番が20pt以上新しいもの
    BORDER=0.85,            # ラインの0.85〜1/0.85倍は「境界」フラグ
)


@dataclass
class Calendar:
    """Factors that put last year's full-year figures on this year's window."""
    start: pd.Timestamp          # current window start (Jan 1)
    end: pd.Timestamp            # current window end (inclusive)
    window_days: int
    w_sales: float               # share of previous-year sales inside the same window
    w_gp: float
    w_units: float
    peak_mult: float             # daily sales of the end month + next month / window daily sales (prev year)
    unit_curve: pd.DataFrame     # prev-year cumulative unit share by day offset, for the implied sell-out date


def derive_calendar(monthly: pd.DataFrame, start: str, end: str) -> Calendar:
    """Build window factors from 月別売上 of the previous calendar year."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if (start_ts.month, start_ts.day) != (1, 1) or end_ts.year != start_ts.year:
        raise ValueError("the current period must start on Jan 1 and end within the same year")
    prev = start_ts.year - 1
    m = monthly[monthly["month"].dt.year == prev]
    m = m.set_index(m["month"].dt.month)
    if len(m) != 12:
        raise ValueError(f"monthly sales for all 12 months of {prev} are required")
    gp = m["純売上(税抜)"] - m["原価"]

    # Fraction of each previous-year month that falls inside the window (partial last month pro-rated).
    frac = pd.Series(0.0, index=range(1, 13))
    frac[frac.index < end_ts.month] = 1.0
    frac[end_ts.month] = end_ts.day / end_ts.days_in_month

    def share(col: pd.Series) -> float:
        return float((col * frac).sum() / col.sum())

    days_prev = pd.Series([pd.Period(f"{prev}-{i:02d}").days_in_month for i in range(1, 13)], index=range(1, 13))
    window_days = (end_ts - start_ts).days + 1
    window_daily = float((m["純売上(税抜)"] * frac).sum()) / window_days
    peak_months = [end_ts.month, end_ts.month % 12 + 1]
    peak_daily = float(m.loc[peak_months, "純売上(税抜)"].sum() / days_prev[peak_months].sum())

    units_in = m["販売点数"] * frac
    days_in = (days_prev * frac).round().astype(int)
    curve = pd.DataFrame({"days": days_in.cumsum(), "cum_share": units_in.cumsum() / units_in.sum()})
    curve = curve[days_in > 0]

    return Calendar(start=start_ts, end=end_ts, window_days=window_days,
                    w_sales=share(m["純売上(税抜)"]), w_gp=share(gp), w_units=share(m["販売点数"]),
                    peak_mult=peak_daily / window_daily, unit_curve=curve)


def implied_sellout_date(q: float, cal: Calendar) -> str:
    """Date by which this year's units would equal last year's pace (t*).

    q = this year's units / last year's units over the same window. If the product
    actually ran out of stock before this date, it was selling at least as well as last year.
    """
    if not np.isfinite(q):
        return ""
    if q >= 1:
        return "期間末以降(需要減の兆候なし)"
    if q <= 0:
        return "前年中に完売"
    days = np.concatenate([[0], cal.unit_curve["days"].to_numpy()])
    cum = np.concatenate([[0.0], cal.unit_curve["cum_share"].to_numpy()])
    d = float(np.interp(q, cum, days))
    return (cal.start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")


def resolve_params(params: dict | None, cal: Calendar, sk: pd.DataFrame) -> dict:
    """Fill the parameters left as None with values measured from the data."""
    P = dict(DEFAULT_PARAMS)
    P.update(params or {})
    if P["G_OWN"] is None:
        own = sk[sk["dept"] != "ETC1"]
        P["G_OWN"] = float(own["gp_26"].sum() / (own["gp_25"].sum() * cal.w_gp))
    if P["PEAK_MULT"] is None:
        P["PEAK_MULT"] = cal.peak_mult
    return P
