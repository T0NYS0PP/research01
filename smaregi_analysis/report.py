"""Build a self-contained HTML report from the analysis outputs.

The page embeds its data as JSON and renders tables/charts in the browser, so the
same file works offline and as a published page. It contains business figures:
keep it out of version control.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import taxonomy as tx
from .model import LABEL_ORDER, STOP_LABELS

LABEL_HELP = {
    "再生産": "分岐点ラインを超えて売れているのに、欠品している（即手配）か、ピーク前の在庫が{cover}日分を切っている（追加発注）。在庫は書き出し時点の値。",
    "再生産・復刻（要確認）": "前年は売れていたが欠品で今期の数字が伸びていない。欠品した日を確認して、同型で再生産か刷新版かを決める。",
    "継続・強化": "ラインを超えて在庫も足りている。枠を維持して欠品させない。",
    "アップデート": "需要の土台はあるが勢いが落ちている、または粗利率が低い。デザイン刷新か価格・原価の見直し。",
    "様子見": "新商品・前年後半投入・成長中などで、まだ判定できない。次回の判定（冬物は冬のピーク後）で再評価する。",
    "終売（在庫消化）": "2年続けてラインを下回り、刷新や値上げでも届かない。在庫を売り切ったら作らない。",
    "終売済み": "在庫0でラインも下回る。再生産しない（多くはすでに自然終売）。",
    "他社ブランド別管理": "原価が登録されていないので粗利で比べられない。掛率を登録してから同じ物差しで評価する。",
}


def _f(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return [{c: _f(v) for c, v in zip(cols, row)} for row in df[cols].itertuples(index=False)]


def build_report(out_dir: str | Path, monthly: pd.DataFrame) -> Path:
    out = Path(out_dir)
    st = pd.read_pickle(out / "styles.pkl")
    H = pd.read_csv(out / "break_point_lines.csv", index_col=0)
    ser = pd.read_csv(out / "series.csv", index_col=0)
    brands = pd.read_csv(out / "other_brands.csv", index_col=0)
    dead = pd.read_csv(out / "dead_skus.csv", dtype={"商品コード": str, "style": str})
    meta = json.loads((out / "run_meta.json").read_text())

    st["label"] = st["label"].astype(str)
    own = st[st.grp != tx.OTHER_GROUP]
    active = own[own.u26 > 0]
    total26 = float(st.s26.sum())
    w = meta["w_sales"]

    # ---- headline numbers
    stop_active = active[active.label.isin(STOP_LABELS)]
    remake = st[st.label == "再生産"]
    check = st[st.label == "再生産・復刻（要確認）"]
    new = own[(own.age == "new") & (own.u26 > 0)]
    other = st[st.grp == tx.OTHER_GROUP]
    sens = meta.get("sensitivity", {})
    base = sens.get("基準", {})
    line_variants = [v for k, v in sens.items() if k.startswith(("基準", "ライン"))]
    strict = max(line_variants, key=lambda v: v["stop_share"], default={})
    young = own[(own.age == "young") & (own.u26 > 0)]
    summary = dict(
        own_total=float(own.s26.sum()), stop_n=base.get("stop_n"), stop_share=base.get("stop_share"),
        stop_share_max=strict.get("stop_share"), stop_n_max=strict.get("stop_n"),
        cover_days=meta["params"]["COVER_REORDER"], season_adjusted=meta["params"]["SEASON_HEAVY"] > 1,
        remake_out_n=base.get("remake_out_n"), remake_out_share=base.get("remake_out_share"),
        remake_cover_n=(base.get("remake_n", 0) - base.get("remake_out_n", 0)),
        remake_cover_share=(base.get("remake_share", 0) - base.get("remake_out_share", 0)),
        young_n=int(len(young)), young_share=float(young.s26.sum() / own.s26.sum()),
        n_launch_sample=meta.get("n_launch_sample"), stability_runs=meta.get("stability_runs"),
        start=meta["start"], end=meta["end"], window_days=meta["window_days"],
        total26=total26, own26=float(own.s26.sum()), other26=float(other.s26.sum()),
        n_active=int(len(active)), n_styles=int(len(own)),
        stop_active_n=int(len(stop_active)), stop_active_share=float(stop_active.s26.sum() / total26),
        remake_n=int(len(remake)), remake_share=float(remake.s26.sum() / total26),
        remake_a=int(remake.detail.str.contains("優先A").sum()), remake_stock=float(remake.stock.sum()),
        check_n=int(len(check)), check_g25=float(check.g25.sum()),
        check_prev_only=int((check.status == "prev_only").sum()),
        new_n=int(len(new)), new_share=float(new.s26.sum() / own.s26.sum()),
        own_ratio=float(own.s26.sum() / (own.s25.sum() * w)), other_ratio=float(other.s26.sum() / (other.s25.sum() * w)),
        own_gm25=float(own.g25.sum() / own.s25.sum()), own_gm26=float(own.g26.sum() / own.s26.sum()),
        stop_stock=float(st[st.label == "終売（在庫消化）"].stock.sum()),
        stab80=float((active.stab >= 0.8).mean()) if "stab" in active else None,
    )

    labels = []
    own_total = float(own.s26.sum())
    for lab in LABEL_ORDER:
        d = st[st.label == lab]
        # Own-product labels are shares of own sales; the other-brand row is its share of all sales.
        denom = total26 if lab == "他社ブランド別管理" else own_total
        labels.append(dict(label=lab, n=int(len(d)), n_active=int((d.u26 > 0).sum()), s26=float(d.s26.sum()),
                           share=float(d.s26.sum() / denom), stock=float(d.stock.sum()),
                           help=LABEL_HELP[lab].format(cover=meta["params"]["COVER_REORDER"])))

    gpu = active.groupby("grp")["gpu"].median()
    lines = []
    for g, r in H.iterrows():
        annual = r.H50 / meta["w_gp"]
        lines.append(dict(grp=g, n=int(r.n_launch), basis=r.get("basis", "新商品"), H25=r.H25, H50=r.H50, H75=r.H75, annual=annual,
                          per_month=annual / gpu.get(g, np.nan) / 12 if g in gpu else None,
                          p10=r.get("H50_p10"), p90=r.get("H50_p90"),
                          n_styles=int((active.grp == g).sum())))
    lines.sort(key=lambda x: -x["H50"])

    cols = ["style", "style_name", "grp", "series", "label", "detail", "age", "s25", "s26", "g26", "E_now", "E_prev",
            "H50", "line_ratio", "stock", "cover_peak", "avail", "DI", "t_star", "stab", "border", "successor",
            "direction", "gm26", "u25", "u26"]
    tab = st.reset_index().rename(columns={"index": "style"})
    if "style" not in tab.columns:
        tab = tab.rename(columns={tab.columns[0]: "style"})
    for c in ["stab", "direction"]:
        if c not in tab.columns:
            tab[c] = None
    styles = _records(tab.sort_values("s26", ascending=False), cols)

    m = monthly.copy()
    m["ym"] = m["month"].astype(str)
    monthly_rows = _records(m.rename(columns={"純売上(税抜)": "sales", "取引数": "tx", "客単価": "aov", "販売点数": "units"}),
                            ["ym", "sales", "tx", "aov", "units"])

    ser = ser.reset_index()
    series_rows = _records(ser, ["series", "n_active", "n_prev_only", "s25", "s26", "FI_instock", "n_new_or_young",
                                 "new_hits", "direction"])
    brand_rows = _records(brands.reset_index(), ["brand", "codes", "s25", "s26", "same_window_ratio", "share26"])
    dead_rows = _records(dead.join(st[["label"]], on="style").fillna({"label": ""}),
                         ["style", "name", "color", "size", "今期販売点数", "在庫", "在庫日数", "label"])

    sens_rows = [dict(name=k, **v) for k, v in sens.items()]
    mk_path = out / "marketing.json"
    marketing = json.loads(mk_path.read_text()) if mk_path.exists() else None
    data = dict(summary=summary, labels=labels, lines=lines, styles=styles, monthly=monthly_rows,
                series=series_rows, brands=brand_rows, dead=dead_rows, params=meta["params"], sensitivity=sens_rows, marketing=marketing)
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    page = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=_f).replace("</", "<\\/"))
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path

