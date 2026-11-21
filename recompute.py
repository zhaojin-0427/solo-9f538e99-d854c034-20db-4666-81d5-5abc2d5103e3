#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立离线复算脚本：只凭结果页下载的 JSON 重算全部体长比较结果。

用法：
    python recompute.py lengths-1.json

脚本只依赖 Python 标准库，不需要 Flask、数据库或本项目的其他文件。
JSON 无效时只打印错误并以非零状态退出，不输出堆栈。
"""

import json
import sys
from bisect import bisect_right
from fractions import Fraction
from math import comb


class InvalidPayload(ValueError):
    """JSON 内容不符合体长对照文件的结构要求。"""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_lengths(data, key):
    if key not in data:
        raise InvalidPayload(f"缺少字段 {key!r}。")
    values = data[key]
    if not isinstance(values, list):
        raise InvalidPayload(f"字段 {key!r} 必须是数组。")
    if not values:
        raise InvalidPayload(f"字段 {key!r} 不能为空。")
    for index, value in enumerate(values, start=1):
        if value is None:
            continue
        if not _is_int(value) or value <= 0:
            raise InvalidPayload(
                f"字段 {key!r} 第 {index} 项必须是正整数毫米或 null。"
            )
    return values


def validate_payload(data):
    """校验并返回只含计算所需字段的规范化 JSON 对象。"""
    if not isinstance(data, dict):
        raise InvalidPayload("JSON 顶层必须是对象。")
    round1 = _validate_lengths(data, "round1_lengths")
    round2 = _validate_lengths(data, "round2_lengths")

    n1 = data.get("n1", len(round1))
    n2 = data.get("n2", len(round2))
    if not _is_int(n1) or n1 <= 0:
        raise InvalidPayload("字段 'n1' 必须是正整数。")
    if not _is_int(n2) or n2 <= 0:
        raise InvalidPayload("字段 'n2' 必须是正整数。")
    if n1 != len(round1):
        raise InvalidPayload(
            f"字段 'n1'（{n1}）与 round1_lengths 的长度（{len(round1)}）不一致。"
        )
    if n2 != len(round2):
        raise InvalidPayload(
            f"字段 'n2'（{n2}）与 round2_lengths 的长度（{len(round2)}）不一致。"
        )

    return {"n1": n1, "n2": n2, "round1": round1, "round2": round2}


def fraction_text(value):
    """以精确分数 a/b 显示（已约分；整数写作 n/1）。"""
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def _observed_statistic(round1, round2):
    """实测数据的最大累计比例绝对差及其所在体长。

    返回 (体长, 首轮不超过该值的条数, 次轮不超过该值的条数, 整数化绝对差)。
    整数化绝对差 = |a*n2 - b*n1|，与比例差 D/(n1*n2) 一一对应。
    """
    n1, n2 = len(round1), len(round2)
    sorted1, sorted2 = sorted(round1), sorted(round2)
    best = None
    for value in sorted(set(round1) | set(round2)):
        a = bisect_right(sorted1, value)
        b = bisect_right(sorted2, value)
        diff = a * n2 - b * n1
        score = abs(diff)
        if best is None or score > best[0]:
            best = (score, value, a, b)
    return best[1], best[2], best[3], best[0]


def permutation_fraction(round1, round2, observed):
    """保持两轮样本数，统计所有等可能重分组中最大差 >= 实测值的比例。

    把两轮全部个体按体长排成一列，每个位置决定属于首轮还是次轮。
    等长记录各占一个位置（各计一条），但只有在整组等长记录都安排好之后
    才检查累计差；用动态规划按组推进，不逐个枚举组合。

    返回 (有利重分组数, 全部重分组数, 比例 Fraction)。
    """
    n1, n2 = len(round1), len(round2)
    total_positions = n1 + n2
    ordered = sorted(round1 + round2)
    boundaries = {
        i
        for i in range(1, total_positions + 1)
        if i == total_positions or ordered[i] != ordered[i - 1]
    }

    # dp[a] = 已放置 i 个位置、其中首轮 a 条，且此前各整组检查均未达到
    # 实测值的分组方式数（即最大差严格小于实测值）。
    dp = [0] * (n1 + 1)
    dp[0] = 1
    for i in range(1, total_positions + 1):
        nxt = [0] * (n1 + 1)
        for a in range(n1 + 1):
            ways = dp[a]
            if a:
                ways += dp[a - 1]
            if ways and i in boundaries:
                if abs(a * n2 - (i - a) * n1) >= observed:
                    ways = 0
            nxt[a] = ways
        dp = nxt

    total = comb(total_positions, n1)
    below = dp[n1]
    favorable = total - below
    return favorable, total, Fraction(favorable, total)


def analyze_lengths(round1, round2):
    """根据两轮毫米读数（缺测为 None）计算页面上的全部结果。"""
    n1, n2 = len(round1), len(round2)
    missing1 = sum(1 for value in round1 if value is None)
    missing2 = sum(1 for value in round2 if value is None)
    result = {
        "measured_counts": [n1 - missing1, n2 - missing2],
        "missing_counts": [missing1, missing2],
        "analysis": None,
    }
    if missing1 or missing2:
        return result

    value, a, b, observed = _observed_statistic(round1, round2)
    diff = Fraction(a, n1) - Fraction(b, n2)
    favorable, total, probability = permutation_fraction(
        round1, round2, observed
    )
    result["analysis"] = {
        "at_length_mm": value,
        "round1_at_or_below": a,
        "round2_at_or_below": b,
        # 累计比例保留各轮样本数作分母，便于核对。
        "p1": f"{a}/{n1}",
        "p2": f"{b}/{n2}",
        "diff": fraction_text(diff),
        "abs_diff": fraction_text(abs(diff)),
        "permutation": {
            "favorable": favorable,
            "total": total,
            "fraction": fraction_text(probability),
        },
    }
    return result


def recompute(data):
    """校验 JSON 并返回计算结果。"""
    payload = validate_payload(data)
    return analyze_lengths(payload["round1"], payload["round2"])


def build_report(data):
    """生成与结果页各项对应的纯文本复算报告。"""
    result = recompute(data)
    missing1, missing2 = result["missing_counts"]
    lines = []
    if missing1 or missing2:
        # 有缺测时只报各轮缺测数。
        lines.append(f"首轮缺测：{missing1} 条")
        lines.append(f"次轮缺测：{missing2} 条")
        return "\n".join(lines)

    item = result["analysis"]
    value = item["at_length_mm"]
    permutation = item["permutation"]
    lines.extend(
        [
            f"首轮样本数：{len(data['round1_lengths'])} 条",
            f"次轮样本数：{len(data['round2_lengths'])} 条",
            f"绝对差最大的体长：{value} mm",
            f"首轮累计比例（不超过 {value} mm）：{item['p1']}",
            f"次轮累计比例（不超过 {value} mm）：{item['p2']}",
            f"两轮比例之差（首轮减次轮）：{item['diff']}",
            f"绝对差：{item['abs_diff']}",
            "等可能重分组中最大绝对差不小于实测值的比例："
            f"{permutation['fraction']}"
            f"（{permutation['favorable']} / {permutation['total']} 种重分组）",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("用法：python recompute.py <结果页下载的 JSON 文件>", file=sys.stderr)
        return 2
    path = args[0]
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        print(f"错误：无法读取文件：{exc}", file=sys.stderr)
        return 2
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print("错误：文件编码需为 UTF-8。", file=sys.stderr)
        return 2
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"错误：不是有效的 JSON（{exc.msg}，第 {exc.lineno} 行）。", file=sys.stderr)
        return 1
    try:
        report = build_report(data)
    except InvalidPayload as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
