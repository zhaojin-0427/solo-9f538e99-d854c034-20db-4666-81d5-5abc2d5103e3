# -*- coding: utf-8 -*-
"""标记重捕估算系统的端到端测试（Flask test client + 临时 SQLite）。"""

import importlib
import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SECRET_KEY", "test-secret")
    sys.modules.pop("app", None)
    app_mod = importlib.import_module("app")
    app_mod.app.config["TESTING"] = True
    with app_mod.app.test_client() as c:
        yield c, app_mod


def csv_bytes(text):
    return io.BytesIO(text.encode("utf-8"))


def upload(client, r1, r2):
    return client.post(
        "/upload",
        data={
            "round1_file": (csv_bytes(r1), "round1.csv"),
            "round2_file": (csv_bytes(r2), "round2.csv"),
        },
        content_type="multipart/form-data",
    )


R1 = "tag\nA\nB\nC\nD\n"
R2 = (
    "sample_no,status,candidates\n"
    "s1,有标,A\n"
    "s2,有标,A;B\n"
    "s3,看不清,B;C\n"
    "s4,无标,\n"
    "s5,看不清,\n"
)


def get_est(app_mod):
    conn = app_mod.get_db()
    row = conn.execute("SELECT * FROM estimate WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else None


def test_upload_feasible_and_estimate(client):
    c, app_mod = client
    resp = upload(c, R1, R2)
    assert resp.status_code == 302  # 成功 → 回首页
    est = get_est(app_mod)
    assert est["n1"] == 4 and est["n2"] == 5
    # 有标 2 条必配 → 最少重捕 2；s3 可再配 → 最多重捕 3
    assert est["min_recap"] == 2
    assert est["max_recap"] == 3
    # N = 4*5/m：上限 10.0（m=2），下限 20/3（m=3）
    assert est["est_high"] == pytest.approx(10.0)
    assert est["est_low"] == pytest.approx(20 / 3)
    min_pairs = {p["sample_no"]: p["tag"] for p in json.loads(est["min_matching"])}
    assert min_pairs == {"s1": "A", "s2": "B"}
    max_pairs = {p["sample_no"]: p["tag"] for p in json.loads(est["max_matching"])}
    assert len(max_pairs) == 3 and max_pairs["s1"] == "A" and max_pairs["s2"] == "B"


def test_infeasible_upload_keeps_old_state(client):
    c, app_mod = client
    assert upload(c, R1, R2).status_code == 302
    # 两个有标样本抢同一个候选号 → 霍尔证书：2 个样本、1 个候选号
    bad_r2 = (
        "sample_no,status,candidates\n"
        "x1,有标,A\n"
        "x2,有标,A\n"
        "x3,无标,\n"
    )
    resp = upload(c, R1, bad_r2)
    assert resp.status_code == 422
    body = resp.get_data(as_text=True)
    assert "无可行配对" in body and "x1" in body and "x2" in body
    # 旧记录与估算原样保留
    est = get_est(app_mod)
    assert est["n2"] == 5 and est["min_recap"] == 2
    conn = app_mod.get_db()
    names = [r[0] for r in conn.execute("SELECT sample_no FROM round2")]
    conn.close()
    assert names == ["s1", "s2", "s3", "s4", "s5"]


def test_correct_success_bumps_version_and_saves(client):
    c, app_mod = client
    upload(c, R1, R2)
    # s5 由无候选的"看不清"改为候选 D → 最多重捕变 4，下限变 5.0
    resp = c.post(
        "/correct",
        data={"sample_no": "s5", "version": "1",
              "status": "看不清", "candidates": "D"},
    )
    assert resp.status_code == 302
    est = get_est(app_mod)
    assert est["max_recap"] == 4
    assert est["est_low"] == pytest.approx(5.0)
    conn = app_mod.get_db()
    row = conn.execute(
        "SELECT version, candidates FROM round2 WHERE sample_no='s5'"
    ).fetchone()
    conn.close()
    assert row["version"] == 2
    assert json.loads(row["candidates"]) == ["D"]


def test_correct_stale_version_rejected(client):
    c, app_mod = client
    upload(c, R1, R2)
    c.post("/correct", data={"sample_no": "s5", "version": "1",
                             "status": "看不清", "candidates": "D"})
    # 用旧版号再次提交 → 过期，估算不变
    resp = c.post(
        "/correct",
        data={"sample_no": "s5", "version": "1",
              "status": "无标", "candidates": ""},
    )
    assert resp.status_code == 409
    assert "已过期" in resp.get_data(as_text=True)
    est = get_est(app_mod)
    assert est["max_recap"] == 4  # 仍是第一次更正后的结果


def test_correct_to_infeasible_rejected(client):
    c, app_mod = client
    upload(c, R1, R2)
    # 把 s2 的候选改成只有 A（与 s1 冲突）→ 无解，保留旧估算
    resp = c.post(
        "/correct",
        data={"sample_no": "s2", "version": "1",
              "status": "有标", "candidates": "A"},
    )
    assert resp.status_code == 422
    body = resp.get_data(as_text=True)
    assert "无可行配对" in body and "s1" in body and "s2" in body
    est = get_est(app_mod)
    assert est["min_recap"] == 2 and est["max_recap"] == 3


def test_zero_recapture_marked_unestimable(client):
    c, app_mod = client
    r2 = "sample_no,status,candidates\ns1,无标,\ns2,看不清,\n"
    assert upload(c, R1, r2).status_code == 302
    est = get_est(app_mod)
    assert est["min_recap"] == 0 and est["max_recap"] == 0
    assert est["est_high"] is None and est["est_low"] is None
    resp = c.get("/")
    assert "不可估" in resp.get_data(as_text=True)


def test_persistence_across_restart(client, tmp_path, monkeypatch):
    c, app_mod = client
    upload(c, R1, R2)
    c.post("/correct", data={"sample_no": "s5", "version": "1",
                             "status": "看不清", "candidates": "D"})
    # 模拟重启：同一 DB 路径重新导入应用
    sys.modules.pop("app", None)
    app_mod2 = importlib.import_module("app")
    app_mod2.app.config["TESTING"] = True
    with app_mod2.app.test_client() as c2:
        body = c2.get("/").get_data(as_text=True)
        assert "s5" in body and "当前估算" in body
    est = get_est(app_mod2)
    assert est["max_recap"] == 4 and est["est_low"] == pytest.approx(5.0)


def test_validation_errors(client):
    c, app_mod = client
    # 无标却带候选号
    resp = upload(c, R1, "sample_no,status,candidates\ns1,无标,A\n")
    assert resp.status_code == 400
    assert "无标" in resp.get_data(as_text=True)
    # 候选号不在首轮
    resp = upload(c, R1, "sample_no,status,candidates\ns1,有标,Z\n")
    assert resp.status_code == 400
    assert "不在首轮" in resp.get_data(as_text=True)
    # 超过 300 条
    big = "tag\n" + "".join(f"T{i}\n" for i in range(301))
    resp = upload(c, big, "sample_no,status,candidates\ns1,无标,\n")
    assert resp.status_code == 400
    assert "1~300" in resp.get_data(as_text=True)
    # 首轮标记号重复
    resp = upload(c, "tag\nA\nA\n", "sample_no,status,candidates\ns1,无标,\n")
    assert resp.status_code == 400
    assert "不唯一" in resp.get_data(as_text=True)
    # 全部失败 → 数据库仍为空
    assert get_est(app_mod) is None


def test_max_rows_boundary(client):
    c, app_mod = client
    r1 = "tag\n" + "".join(f"T{i}\n" for i in range(300))
    r2 = "sample_no,status,candidates\n" + "".join(
        f"s{i},有标,T{i}\n" for i in range(300)
    )
    assert upload(c, r1, r2).status_code == 302
    est = get_est(app_mod)
    assert est["min_recap"] == 300 and est["max_recap"] == 300
    assert est["est_high"] == pytest.approx(300.0)


# ---------------------------------------------------------------- 体长对照

R1_LEN = "tag,length_mm\nA,100\nB,110\nC,120\nD,140\n"
R2_LEN = (
    "sample_no,status,candidates,length_mm\n"
    "s1,有标,A,105\n"
    "s2,有标,A;B,115\n"
    "s3,看不清,B;C,125\n"
    "s4,无标,,135\n"
)


def get_lengths(app_mod, seq="current"):
    conn = app_mod.get_db()
    try:
        if seq == "current":
            survey = conn.execute("SELECT * FROM survey WHERE id = 1").fetchone()
            seq = survey["canonical_seq"]
        row = conn.execute(
            "SELECT payload_json FROM submission WHERE seq = ?", (seq,)
        ).fetchone()
    finally:
        conn.close()
    return json.loads(row["payload_json"])


def test_lengths_mm_and_cm_upload(client):
    c, app_mod = client
    r2_cm = (
        "sample_no,status,candidates,length_cm\n"
        "s1,有标,A,10.5\n"
        "s2,有标,A;B,11.5\n"
        "s3,看不清,B;C,12.5\n"
        "s4,无标,,\n"
    )
    assert upload(c, R1_LEN, r2_cm).status_code == 302
    payload = get_lengths(app_mod)
    assert payload["round1_lengths"] == [100, 110, 120, 140]
    assert payload["round2_lengths"] == [105, 115, 125, None]


def test_length_rejections_keep_previous_state(client):
    c, app_mod = client
    assert upload(c, R1_LEN, R2_LEN).status_code == 302
    before = get_lengths(app_mod)
    bad_cases = [
        # 两列并存（即便都留空）
        "tag,length_mm,length_cm\nA,100,\nB,,\n",
        # 厘米无法精确换成正整数毫米
        "tag,length_cm\nA,10.25\n",
        # 毫米不是正整数
        "tag,length_mm\nA,10.5\n",
        "tag,length_mm\nA,-3\n",
        "tag,length_mm\nA,0\n",
        # 次轮同样拒收
        "sample_no,status,candidates,length_cm\ns1,无标,,10.05\n",
    ]
    for text in bad_cases:
        if text.startswith("sample_no"):
            resp = upload(c, R1_LEN, text)
        else:
            resp = upload(c, text, "sample_no,status,candidates\ns1,无标,\n")
        assert resp.status_code == 400, text
        assert "整份不予接收" in resp.get_data(as_text=True)
    # 全部拒收 → 旧数据与快照原样保留
    assert get_lengths(app_mod) == before
    est = get_est(app_mod)
    assert est["n2"] == 4


def test_compare_page_full_data_and_download(client):
    c, app_mod = client
    assert upload(c, R1_LEN, R2_LEN).status_code == 302
    body = c.get("/compare").get_data(as_text=True)
    assert "体长（并列取最小）" in body
    assert "100 mm" in body
    # 首轮 <=100 有 1/4，次轮 <=100 有 0/4，差 1/4
    assert "1/4" in body and "0/4" in body
    # 全部 70 种重分组都至少达到实测差
    assert "70" in body and "1/1" in body

    resp = c.get("/lengths/current.json")
    assert resp.status_code == 200
    data = json.loads(resp.get_data(as_text=True))
    assert data["round1_lengths"] == [100, 110, 120, 140]
    assert data["round2_lengths"] == [105, 115, 125, 135]
    assert resp.headers["Content-Disposition"].endswith('.json"')


def test_compare_page_only_reports_missing_counts(client):
    c, app_mod = client
    r1 = "tag,length_mm\nA,100\nB,\n"
    r2 = "sample_no,status,candidates,length_mm\ns1,无标,,105\ns2,无标,,\n"
    assert upload(c, r1, r2).status_code == 302
    body = c.get("/compare").get_data(as_text=True)
    assert "首轮缺测 1 条" in body and "次轮缺测 1 条" in body
    assert "2. 实测累计比例的最大绝对差" not in body
    assert "3. 重分组检验" not in body
    payload = get_lengths(app_mod)
    assert payload["round1_lengths"] == [100, None]
    assert payload["round2_lengths"] == [105, None]


def test_correct_length_mm_or_clear(client):
    c, app_mod = client
    assert upload(c, R1_LEN, R2_LEN).status_code == 302
    # 按毫米更正次轮记录 s4
    resp = c.post(
        "/correct",
        data={"sample_no": "s4", "version": "1", "status": "无标",
              "candidates": "", "length_mm": "150"},
    )
    assert resp.status_code == 302
    payload = get_lengths(app_mod)
    assert payload["round2_lengths"] == [105, 115, 125, 150]
    # 再清空 s4 体长，随记录提交
    resp = c.post(
        "/correct",
        data={"sample_no": "s4", "version": "2", "status": "无标",
              "candidates": "", "length_mm": ""},
    )
    assert resp.status_code == 302
    assert get_lengths(app_mod)["round2_lengths"] == [105, 115, 125, None]
    # 非法体长 → 拒绝，记录保留
    resp = c.post(
        "/correct",
        data={"sample_no": "s4", "version": "3", "status": "无标",
              "candidates": "", "length_mm": "15.5"},
    )
    assert resp.status_code == 400
    assert "正整数毫米" in resp.get_data(as_text=True)
    conn = app_mod.get_db()
    row = conn.execute(
        "SELECT version, length_mm FROM round2 WHERE sample_no='s4'"
    ).fetchone()
    conn.close()
    assert row["version"] == 3 and row["length_mm"] is None


def test_old_survey_download_survives_correction_and_reupload(client):
    c, app_mod = client
    assert upload(c, R1_LEN, R2_LEN).status_code == 302
    old_text = c.get("/lengths/1.json").get_data(as_text=True)
    # 更正产生第 2 次提交
    assert c.post(
        "/correct",
        data={"sample_no": "s4", "version": "1", "status": "无标",
              "candidates": "", "length_mm": "150"},
    ).status_code == 302
    assert get_lengths(app_mod)["round2_lengths"] == [105, 115, 125, 150]
    # 整份重传产生第 3 次提交，且同号只取新数据
    r1_new = "tag,length_mm\nA,100\nB,110\nC,120\nD,140\n"
    r2_new = "sample_no,status,candidates,length_mm\ns1,无标,,999\n"
    assert upload(c, r1_new, r2_new).status_code == 302
    payload = get_lengths(app_mod)
    assert payload["round2_lengths"] == [999]
    assert payload["n2"] == 1
    # 旧编号下载内容不变
    assert c.get("/lengths/1.json").get_data(as_text=True) == old_text
    old = json.loads(old_text)
    assert old["round2_lengths"] == [105, 115, 125, 135]
    # 历史列表仍能看到旧提交
    body = c.get("/compare").get_data(as_text=True)
    assert "lengths-1.json" in body and "lengths-3.json" in body


def test_offline_recompute_matches_page(client, tmp_path):
    c, app_mod = client
    assert upload(c, R1_LEN, R2_LEN).status_code == 302
    json_text = c.get("/lengths/current.json").get_data(as_text=True)
    path = tmp_path / "lengths-1.json"
    path.write_text(json_text, encoding="utf-8")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import recompute

    data = json.loads(json_text)
    report = recompute.build_report(data)
    body = c.get("/compare").get_data(as_text=True)
    assert "100 mm" in report and "100 mm" in body
    assert "1/4" in report and "0/4" in report
    assert "70" in report and "1/1" in report
    assert report == recompute.build_report(recompute.validate_payload(data) | {
        "n1": data["n1"], "n2": data["n2"],
        "round1_lengths": data["round1_lengths"],
        "round2_lengths": data["round2_lengths"],
    })


def test_recompute_rejects_invalid_files(tmp_path):
    import subprocess

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recompute.py")
    valid = {
        "n1": 2, "n2": 2,
        "round1_lengths": [10, 20], "round2_lengths": [15, None],
    }
    cases = {
        "not-json.json": "{oops",
        "array.json": "[]",
        "missing.json": json.dumps({"n1": 1}),
        "badlen.json": json.dumps(
            {"n1": 3, "n2": 2, "round1_lengths": [1, 2],
             "round2_lengths": [1, 2]}
        ),
    }
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-I", script, str(path)],
            capture_output=True, text=True,
        )
        assert proc.returncode != 0, name
        assert proc.stdout == ""
        assert proc.stderr.startswith("错误：")
        assert "Traceback" not in proc.stderr

    # 有效的缺测文件只报缺测数
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", script, str(path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip().splitlines() == ["首轮缺测：0 条", "次轮缺测：1 条"]


def test_permutation_matches_brute_force():
    import itertools
    from fractions import Fraction

    from recompute import analyze_lengths

    def brute(round1, round2):
        n1, n2 = len(round1), len(round2)
        tagged = sorted(round1)
        values = sorted(set(round1) | set(round2))

        def stat(first):
            second = sorted(round2 + [x for x in round1 if x not in first])
            best = 0
            for v in values:
                a = sum(1 for x in first if x <= v)
                b = sum(1 for x in second if x <= v)
                best = max(best, abs(a * n2 - b * n1))
            return best

        observed = stat(list(round1))
        total = 0
        favorable = 0
        pool = round1 + round2
        for combo in itertools.combinations(range(n1 + n2), n1):
            total += 1
            if stat([pool[i] for i in combo]) >= observed:
                favorable += 1
        return Fraction(favorable, total)

    cases = [
        ([10, 20], [15, 25]),
        ([10], [20]),
        ([5, 5, 9], [5, 7]),
        ([1, 2, 2], [2, 3]),
    ]
    for round1, round2 in cases:
        result = analyze_lengths(round1, round2)["analysis"]["permutation"]
        expected = brute(round1, round2)
        assert result["fraction"] == f"{expected.numerator}/{expected.denominator}"

