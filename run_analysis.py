"""Run the stop-selling / update / new-product analysis on Smaregi exports.

Example:
    python3 run_analysis.py \
        --prev  商品別売上_20250101-20251231.csv \
        --curr  商品別売上_20260101-20261004.csv --curr-start 2026-01-01 --curr-end 2026-10-04 \
        --monthly 月別売上_202510.csv 月別売上_202610.csv \
        --out output/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smaregi_analysis.loaders import load_monthly_sales, load_product_sales
from smaregi_analysis.model import (LABEL_ORDER, brand_table, build_sku_table, compute_styles, dead_skus,
                                    series_table, stability)
from smaregi_analysis.params import derive_calendar, resolve_params
from smaregi_analysis.report import build_report

STYLE_COLUMNS = {
    "style_name": "商品名", "dept": "部門", "grp": "棚グループ", "series": "シリーズ", "label": "判定",
    "detail": "理由・アクション", "rule": "ルール", "border": "境界", "stab": "安定度",
    "age": "ライフステージ", "s25": "前年売上", "s26": "今期売上", "g25": "前年粗利", "g26": "今期粗利",
    "u25": "前年点数", "u26": "今期点数", "stock": "在庫", "avail": "有効在庫率", "cover_peak": "在庫日数(ピーク換算)",
    "E_now": "判定粗利(今期)", "E_prev": "前年粗利(同期間換算)", "H25": "ライン下限H25", "H50": "分岐点ラインH50",
    "H75": "優先ラインH75", "line_ratio": "ライン比", "DI": "前年比", "TI": "棚内トレンド", "gm26": "粗利率",
    "gpu": "1点粗利", "t_star": "仮想欠品日", "successor": "後継品番", "direction": "空き枠の行き先",
    "dead_sku": "死に筋SKU数", "dead_stock": "死に筋在庫",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prev", required=True, help="前年1年分の商品別売上CSV")
    ap.add_argument("--curr", required=True, help="今年1/1〜の商品別売上CSV")
    ap.add_argument("--curr-start", required=True)
    ap.add_argument("--curr-end", required=True)
    ap.add_argument("--monthly", nargs="+", required=True, help="前年1〜12月を含む月別売上CSV")
    ap.add_argument("--out", default="output")
    ap.add_argument("--params", help="上書きするパラメータのJSONファイル")
    ap.add_argument("--stability-runs", type=int, default=200)
    ap.add_argument("--report-only", action="store_true", help="既存の出力からレポートだけ作り直す")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly_sales(args.monthly)
    if args.report_only:
        print(f"wrote {build_report(out, monthly).resolve()}")
        return
    cal = derive_calendar(monthly, args.curr_start, args.curr_end)
    sk = build_sku_table(load_product_sales(args.prev), load_product_sales(args.curr))
    P = resolve_params(json.loads(Path(args.params).read_text()) if args.params else None, cal, sk)

    st, H, launches = compute_styles(sk, P, cal)
    ser = series_table(st)
    st = st.join(ser[["direction"]], on="series")
    st["direction"] = st["direction"].fillna("(単独)：新コンセプト or 店主判断")
    if args.stability_runs:
        stab, line_band = stability(sk, P, cal, st, args.stability_runs)
        st = st.join(stab)
        H = H.join(line_band.rename(columns={0.1: "H50_p10", 0.5: "H50_p50", 0.9: "H50_p90"}))

    st["label"] = pd.Categorical(st["label"], LABEL_ORDER, ordered=True)
    table = st.sort_values(["label", "s26"], ascending=[True, False])
    table = table[[c for c in STYLE_COLUMNS if c in table.columns]].rename(columns=STYLE_COLUMNS)
    table.index.name = "品番"
    table.to_csv(out / "style_decisions.csv", encoding="utf-8-sig")
    dead_skus(sk, st, P, cal).to_csv(out / "dead_skus.csv", encoding="utf-8-sig", index=False)
    ser.to_csv(out / "series.csv", encoding="utf-8-sig")
    brand_table(st, cal).to_csv(out / "other_brands.csv", encoding="utf-8-sig")
    H.to_csv(out / "break_point_lines.csv", encoding="utf-8-sig")
    st.to_pickle(out / "styles.pkl")
    meta = dict(params=P, window_days=cal.window_days, w_sales=cal.w_sales, w_gp=cal.w_gp, w_units=cal.w_units,
                peak_mult=cal.peak_mult, start=str(cal.start.date()), end=str(cal.end.date()), n_launch_sample=len(launches))
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float))

    counts = st.groupby("label", observed=True).agg(styles=("label", "size"), sales=("s26", "sum"))
    counts["sales_share"] = counts["sales"] / st["s26"].sum()
    print(counts.round(3).to_string())
    print(f"\nwrote {out.resolve()} (report: {build_report(out, monthly).name})")


if __name__ == "__main__":
    main()
