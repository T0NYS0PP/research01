"""Shop-specific grouping rules: shelf groups, product types, design series, collaborations.

These are keyword rules over Smaregi 部門名 and 商品名. Edit them when new departments,
series or collaborations are added.
"""
from __future__ import annotations

import re

# 部門 -> 棚グループ (products on the same shelf compete for the same slot).
DEPT_GROUP = {
    "ｶｯﾄｿｰB:SHORT": "Tシャツ", "LDｶｯﾄｿｰB:SHORT": "Tシャツ", "ｼｬﾂB:SHORT": "Tシャツ",
    "ｶｯﾄｿｰA:LONG": "スウェット・フーディ", "ｼｬﾂA:LONG": "スウェット・フーディ",
    "ｼﾞｬｹｯﾄ": "アウター", "LDｼﾞｬｹｯﾄ": "アウター", "ｺｰﾄ": "アウター",
    "ﾊﾟﾝﾂA:LONG": "パンツ", "ﾊﾟﾝﾂB:SHORT": "パンツ", "ｷｯｽﾞ": "キッズ",
    "ｷｬｯﾌﾟ": "帽子・バッグ・靴", "ﾊﾞｯｸﾞ": "帽子・バッグ・靴", "ｼｭｰｽﾞ": "帽子・バッグ・靴",
    "ｱｸｾｻﾘｰ": "小物", "ﾊﾟｯﾁ･ﾋﾟﾝｽﾞ": "小物", "ﾎﾞﾃﾞｨ": "小物", "イベント": "小物",
    "ETC1": "他社",
}
GOODS_GROUPS = {"帽子・バッグ・靴", "小物"}
OTHER_GROUP = "他社"

OUTER_KW = r"HANTEN|半纏|SAMUE|作務衣|SUKAJAN|ｽｶｼﾞｬﾝ|JACKET| JK|BOMBER|COAT|ｺｰﾄ|RIKYU"
NOT_OUTER_KW = r"HOODIE|HANTEN HO|LSｼｬﾂ|ｼｬﾂ|CARDIGAN"
HEAVY_KW = r"HOODIE|HANTEN HO|ﾌｰﾃﾞｨ|HEAVY|KNIT|ﾆｯﾄ|SWEAT|ｽｳｪｯﾄ|CHENILLE|CARDIGAN|ﾊﾟｰｶ| PO |SalamanderSW|\bSW\b"
LSTEE_KW = r"L/S|LS TEE|LS T\b|HEAVY LS|ﾛﾝT|LONG ?T"
NOT_LSTEE_KW = r"HOODIE|CARDIGAN|KNIT|ｼｬﾂ"
LIMITED_KW = r"Limited|Ltd|限定"

COLLAB_KW = (r"GUNDAM|Zaku|EVA\+|NARUTO|SASUKE|ナルト|サスケ|ULTRA|ｳﾙﾄﾗ|鬼太郎|目玉の親父|炎炎|新門|森羅|ｱｰｻｰ|"
             r"聖陽十字|攻殻|BEBOP|初音|ｺﾞｼﾞﾗ|DYC|BADMEAW|だるま商店|×|LAUREN|KATSUMATA|Nychos|CHiNPAN|MILTZ|坩堝")

SERIES = [
    ("桜花爛漫", r"桜花爛漫"), ("江戸東京", r"江戸東京"), ("浅柴", r"浅柴"), ("英姿颯爽", r"英姿颯爽"),
    ("だるま商店", r"だるま商店"), ("KIMO NO TEE", r"KIMO ?NO ?T|KIMONOTEE|KIMONO ?T"), ("NEON", r"NEON"),
    ("ゴジラ", r"ｺﾞｼﾞﾗ"), ("ガンダム", r"GUNDAM|Zaku"), ("エヴァ", r"EVA\+"),
    ("ナルト", r"NARUTO|SASUKE|ナルト|サスケ|纏NARUTO"), ("ウルトラマン", r"ULTRA|ｳﾙﾄﾗ"),
    ("鬼太郎", r"鬼太郎|目玉の親父"), ("炎炎ノ消防隊", r"炎炎|新門|森羅|ｱｰｻｰ|聖陽十字"), ("攻殻機動隊", r"攻殻"),
    ("BADMEAW", r"BADMEAW"), ("紋シリーズ", r"PigmentDyed|紋 ?CAP|紋ﾊﾞｹｯﾄ"), ("鯢百態", r"鯢百態"),
    ("火消之図", r"火消之図"), ("抜染スラブ", r"抜染|ｽﾗﾌﾞ|SLAB"), ("HANTEN HOODIE", r"HANTEN HO"),
    ("RIKYU", r"RIKYU"), ("作務衣・ダボ", r"SAMUE|DABO"), ("火喰鳥", r"火喰鳥"), ("火消分隊", r"火消分隊"),
    ("OTO", r"^OTO|^おと"), ("鬼楊柳", r"鬼楊柳"), ("DENIM半纏", r"DENIM"), ("TradPatt", r"TradPatt"),
    ("浮世絵REMIX", r"浮世絵REMIX"), ("PATCH", r"PATCH"),
    ("香・バーム", r"LUTEN|JINKO|LAVENDER|BEYOND|TREE OF LIFE|ZUKO|BALM|SACHET"),
    ("DYC", r"DYC"), ("SANGOU", r"SANGOU"),
    ("四字熟語T", r"寿山福海|多幸多福|喜怒哀楽|歓喜踊躍|祝着至極|七転八起|一期一会"),
]
NO_SERIES = "(単独)"

# Words too common to link a product to its successor: colours, garment types, sizes,
# fabrics/finishes and collaboration partners (a DYC jacket does not replace a DYC samue).
GENERIC_TOKENS = set(
    """TEE T TANK LS L S SS PO HOODIE ZIP HEAVY HANTEN PANTS WIDE SHORT SHORTS CAP JK JACKET SAMUE DABO KD NEON KIMO NO
    BLACK WHITE NAVY RED BLUE GREEN PINK BEIGE OLIVE GRAY GREY BROWN KHAKI CHARCOAL VANILLA DENIM BAG TOTE PATCH UV SLAB VER
    M XL XS XXL HIKESHI THE OF X BK WH NV GR DE RS LD SW KN KNIT PT BIG SET NEW
    BORO JACQUARD SASHIKO WASH PLAID OPAL RUSTED BOKEH ASA LINEN CHENILLE GLOW REFLECT THERMO PIGMENTDYED
    DYC GUNDAM ZAKU EVA ULTRA NARUTO SASUKE LAUREN KATSUMATA MILTZ CHINPAN NYCHOS BADMEAW NOPE LUTEN""".split()
) | {"半纏", "火消", "火消魂", "ﾌﾞﾗｯｸ", "ﾎﾜｲﾄ", "ﾈｲﾋﾞｰ", "ﾚｯﾄﾞ", "ｶｰｷ", "ｸﾞﾘｰﾝ", "ﾋﾟﾝｸ", "ｸﾞﾚｰ", "ﾅﾁｭﾗﾙ",
     "ｲｴﾛｰ", "ｽﾗﾌﾞ", "ﾃﾞﾆﾑ", "ﾊﾞｹｯﾄﾊｯﾄ", "ｺﾞｼﾞﾗ", "ｳﾙﾄﾗ"}


def series_of(name: str) -> str:
    n = re.sub(r"^KD\s*", "", str(name))
    for label, pattern in SERIES:
        if re.search(pattern, n):
            return label
    return NO_SERIES


def name_tokens(core: str) -> set[str]:
    toks = re.findall(r"[A-Za-z]+|[^\x00-\x7F]+", str(core))
    return {t.upper() for t in toks if len(t) >= 2 and t.upper() not in GENERIC_TOKENS}


def compact_name(core: str) -> str:
    """Name without spaces and NEW/N prefixes, to catch re-registrations like 'LAVENDER ME' -> 'NEW LUTEN LAVENDERME'."""
    n = re.sub(r"^(NEW|N)\s+", "", str(core).upper().strip())
    return re.sub(r"[\s\-_()（）]", "", n)
