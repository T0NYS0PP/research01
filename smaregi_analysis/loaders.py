"""Read Smaregi CSV exports (商品別売上 / 月別売上) into tidy DataFrames.

Smaregi admin CSV exports are Shift_JIS (CP932); UTF-8 copies are accepted too.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Departments that are not merchandise (shopping bags, shipping fees, test codes).
NON_PRODUCT_DEPTS = {"ETC2", "DRINK/FOOD", "未登録部門"}
# Department for other brands registered as one code per brand with no cost price.
OTHER_BRAND_DEPT = "ETC1"

_STR_COLS = {"商品コード": str, "品番": str, "グループコード": str, "カラー": str, "サイズ": str}


def read_smaregi_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a Smaregi CSV, trying CP932 first and falling back to UTF-8."""
    for enc in ("cp932", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path} as CP932 or UTF-8")


def load_product_sales(path: str | Path) -> pd.DataFrame:
    """Load a 商品別売上 export: one row per SKU, total row removed.

    Duplicate 商品コード rows (same code re-registered under a slightly
    different name) are summed so every SKU appears once.
    """
    df = read_smaregi_csv(path, dtype=_STR_COLS)
    df = df[df["商品コード"] != "合計"].copy()
    # Sales rung up against a department without a product (部門販売) have no code.
    no_code = df["商品コード"].isna()
    df.loc[no_code, "商品コード"] = "部門販売:" + df.loc[no_code, "部門名"].fillna("")
    num_cols = ["純売上", "純売上(税抜)", "消費税", "原価", "販売点数", "返品数", "在庫数"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ("品番", "グループコード"):
        df[c] = df[c].fillna(df["商品コード"])
    df["カラー"] = df["カラー"].fillna("-")
    df["サイズ"] = df["サイズ"].fillna("-")

    agg = {c: "sum" for c in num_cols if c != "在庫数"}
    # Stock is a point-in-time snapshot; duplicates of one code share the same value.
    agg.update({"在庫数": "max", "商品名": "first", "部門名": "first", "品番": "first",
                "カラー": "first", "サイズ": "first", "グループコード": "first"})
    df = df.groupby("商品コード", as_index=False).agg(agg)
    df["style_name"] = [style_name(n, c, s) for n, c, s in zip(df["商品名"], df["カラー"], df["サイズ"])]
    df["color_norm"] = df["カラー"].map(normalize_color)
    df["kind"] = df["部門名"].map(product_kind)
    df.loc[df["商品コード"].str.startswith("部門販売:"), "kind"] = "non_product"
    return df


def load_monthly_sales(paths: list[str | Path]) -> pd.DataFrame:
    """Load one or more 月別売上 exports, drop total rows and de-duplicate months."""
    frames = [read_smaregi_csv(p, dtype={"日付": str}) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["日付"].str.match(r"^\d{4}/\d{2}$", na=False)].drop_duplicates("日付", keep="last")
    df["month"] = pd.PeriodIndex(df["日付"].str.replace("/", "-"), freq="M")
    for c in df.columns:
        if c not in ("日付", "month"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("month").reset_index(drop=True)


def product_kind(dept: str) -> str:
    if dept in NON_PRODUCT_DEPTS:
        return "non_product"
    if dept == OTHER_BRAND_DEPT:
        return "other_brand"
    return "own"


def style_name(name: str, color: str, size: str) -> str:
    """Strip the trailing ' <カラー> <サイズ>' from a SKU name to get the design name."""
    name = str(name).strip()
    for token in (size, color):
        token = str(token).strip()
        if token and name.endswith(" " + token):
            name = name[: -(len(token) + 1)].rstrip()
        elif token == "-" and name.endswith("-"):
            name = name[:-1].rstrip()
    return name


_COLOR_ALIASES = {
    "ﾌﾞﾗｯｸ": "BLACK", "ﾎﾜｲﾄ": "WHITE", "ﾈｲﾋﾞｰ": "NAVY", "ｵﾘｰﾌﾞ": "OLIVE", "ｸﾞﾘｰﾝ": "GREEN",
    "ﾚｯﾄﾞ": "RED", "ﾋﾟﾝｸ": "PINK", "ｸﾞﾚｰ": "GRAY", "GREY": "GRAY", "ﾅﾁｭﾗﾙ": "NATURAL",
    "ﾊﾞｰｶﾞﾝﾃﾞｨ": "BURGUNDY", "ﾌﾞﾙｰ": "BLUE", "ﾍﾞｰｼﾞｭ": "BEIGE", "ｶｰｷ": "KHAKI", "ｲｴﾛｰ": "YELLOW",
    "ﾌﾞﾗｳﾝ": "BROWN", "ｵﾚﾝｼﾞ": "ORANGE", "ﾊﾟｰﾌﾟﾙ": "PURPLE", "ﾁｬｺｰﾙ": "CHARCOAL", "ｻｸﾗ": "SAKURA",
}


def normalize_color(color: str) -> str:
    c = re.sub(r"\s+", "", str(color)).upper()
    return _COLOR_ALIASES.get(str(color).strip(), c)
