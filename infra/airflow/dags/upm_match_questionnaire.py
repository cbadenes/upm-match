"""UPM Match — questionnaire (PDI + student) analysis DAG.

Downloads the evaluation-questionnaire responses that the web app stores in MinIO (bucket `MINIO_EVAL_BUCKET` under `responses/<role>/<date>/*.json`)
and produces aggregated, publication-ready analyses (one for PDIs and one for students) that are written back to MinIO under `analysis/<run_ts>/` (and mirrored to `analysis/latest/`).

Tasks
-----
1. `download_responses` — pull every response JSON, split by role and stage the raw consolidated sets in MinIO for the analysis tasks.
2. `analyze_pdi` — aggregate the PDI responses (Likert stats with reverse-coding by polarity, knowledge-domain relevance levels, recommendation-case metrics, open text, completion/timing/device quality). Writes `pdi_analysis.json` + `pdi_report.md`.
3. `analyze_students` — the equivalent for students (context distributions, Likert, multi/single choices, open text, quality). Writes `student_analysis.json` + `student_report.md`.
4. `combine_report` *(added extra)* — merge both into a single `report.md` + `summary.json` and log a concise overview in the task log.
5. `fetch_to_repo` — mirror `analysis/latest/` into the mounted repo checkout (`evaluation/reports/questionnaire/`) and echo both reports into the task log, since MinIO has no public route on the server.
"""

from __future__ import annotations

from datetime import datetime

from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.python import ExternalPythonOperator

# Interpreter that has the project dependencies (built by infra/airflow/Dockerfile).
_VENV_PYTHON = "/home/airflow/upm_venv/bin/python"


def run_stage(stage: str, run_ts: str, since: str = "") -> dict:
    """Single self-contained entry point for every task (branch on `stage`).

    Runs in the project virtualenv, so it may import `minio`.
    All helpers are nested so the whole callable is shipped to the external interpreter as one unit.

    :param stage: one of "download", "pdi", "student", "combine".
    :param run_ts: run timestamp (`ts_nodash`) used to namespace the outputs in MinIO.
    :param since: optional `YYYY-MM-DD`; ignore responses submitted before it.
    :return: a small dict summary (also pushed to XCom).
    """
    import io
    import json
    import os
    import re
    import statistics
    from collections import Counter
    from datetime import datetime, timezone

    from minio import Minio

    # MinIO plumbing
    bucket = os.environ.get("MINIO_EVAL_BUCKET", "upm-match-evaluations")
    client = Minio(
        f"{os.environ.get('MINIO_HOST', 'minio')}:{os.environ.get('MINIO_PORT', '9000')}",
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=False,
    )
    analysis_prefix = f"analysis/{run_ts}"

    def read_json(name):
        resp = client.get_object(bucket, name)
        try:
            return json.loads(resp.read().decode("utf-8"))
        finally:
            resp.close()
            resp.release_conn()

    def read_text(name):
        resp = client.get_object(bucket, name)
        try:
            return resp.read().decode("utf-8")
        finally:
            resp.close()
            resp.release_conn()

    def write_bytes(name, data, content_type):
        client.put_object(
            bucket, name, io.BytesIO(data), length=len(data), content_type=content_type
        )

    def write_json(name, payload):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        write_bytes(name, data, "application/json")

    def write_text(name, text):
        write_bytes(name, text.encode("utf-8"), "text/markdown; charset=utf-8")

    def load_responses(role):
        """Read every `responses/<role>/**.json` (optionally filtered by date)."""
        out = []
        prefix = f"responses/{role}/"
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            if not obj.object_name.endswith(".json"):
                continue
            try:
                rec = read_json(obj.object_name)
            except Exception:  # noqa: BLE001 — skip an unreadable/partial object
                continue
            if since:
                d = (rec.get("metadata") or {}).get("submitted_date") or ""
                if d and d < since:
                    continue
            out.append(rec)
        return out

    def iter_dicts(obj):
        if isinstance(obj, dict):
            yield obj
            for v in obj.values():
                yield from iter_dicts(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from iter_dicts(v)

    def norm_id(qid):
        # Collapse per-case (s5_<hex>_) and per-year (s3_<year>_) prefixes so the same question aggregates across cases/years regardless of the run.
        qid = re.sub(r"^s5_[0-9a-fA-F]{4,}_", "s5_", qid)
        qid = re.sub(r"^s3_\d{4}_", "s3_", qid)
        return qid

    def summary_stats(values):
        n = len(values)
        if not n:
            return {"n": 0, "mean": None, "sd": None, "median": None}
        return {
            "n": n,
            "mean": round(statistics.mean(values), 3),
            "sd": round(statistics.stdev(values), 3) if n > 1 else 0.0,
            "median": statistics.median(values),
        }

    def likert_table(records):
        """Aggregate every Likert answer found anywhere, grouped by (normalised) id."""
        groups = {}
        for rec in records:
            for d in iter_dicts(rec.get("answers")):
                if "polarity" in d and "value" in d and "id" in d:
                    key = norm_id(d["id"])
                    g = groups.setdefault(
                        key,
                        {
                            "text": d.get("text"),
                            "polarity": d.get("polarity"),
                            "scale": d.get("scale", "1-5"),
                            "values": [],
                        },
                    )
                    if d.get("text"):
                        g["text"] = d["text"]
                    if isinstance(d.get("value"), int):
                        g["values"].append(d["value"])
        table = {}
        for key, g in groups.items():
            stats = summary_stats(g["values"])
            entry = {
                "text": g["text"],
                "polarity": g["polarity"],
                "scale": g["scale"],
                "distribution": dict(sorted(Counter(g["values"]).items())),
                **stats,
            }
            # Reverse-coded mean for negatively worded items (higher raw = worse).
            if g["polarity"] == "negative" and stats["mean"] is not None:
                entry["mean_reversed"] = round(6 - stats["mean"], 3)
            table[key] = entry
        return table

    def open_table(records):
        groups = {}
        for rec in records:
            for d in iter_dicts(rec.get("answers")):
                if (
                    "id" in d
                    and isinstance(d.get("value"), str)
                    and "polarity" not in d
                    and "options" not in d
                    and "level" not in d
                    and "keyword" not in d
                ):
                    key = norm_id(d["id"])
                    g = groups.setdefault(key, {"text": d.get("text"), "answers": []})
                    if d.get("text"):
                        g["text"] = d["text"]
                    val = (d.get("value") or "").strip()
                    if val:
                        g["answers"].append(val)
        return {
            key: {
                "text": g["text"],
                "answered": len(g["answers"]),
                "answers": g["answers"],
            }
            for key, g in groups.items()
        }

    def meta_quality(records):
        n = len(records)
        devices, langs, versions, dates = Counter(), Counter(), Counter(), Counter()
        durations, likert_rates, open_rates = [], [], []
        for rec in records:
            m = rec.get("metadata") or {}
            devices[m.get("device") or "unknown"] += 1
            langs[m.get("language") or "unknown"] += 1
            versions[m.get("form_version") or "unknown"] += 1
            if m.get("submitted_date"):
                dates[m["submitted_date"]] += 1
            if isinstance(m.get("duration_seconds"), int):
                durations.append(m["duration_seconds"])
            rs = m.get("response_stats") or {}
            if rs.get("likert_total"):
                likert_rates.append(rs["likert_answered"] / rs["likert_total"])
            if rs.get("open_total"):
                open_rates.append(rs["open_answered"] / rs["open_total"])
        return {
            "n_responses": n,
            "by_form_version": dict(versions),
            "by_device": dict(devices),
            "by_language": dict(langs),
            "by_date": dict(sorted(dates.items())),
            "duration_seconds": summary_stats(durations),
            "mean_likert_completion": round(statistics.mean(likert_rates), 3)
            if likert_rates
            else None,
            "mean_open_completion": round(statistics.mean(open_rates), 3)
            if open_rates
            else None,
        }

    # PDI-specific
    def pdi_keywords(records):
        levels = Counter()
        level_values, added = [], Counter()
        kept_true = kept_total = 0
        for rec in records:
            kw = (rec.get("answers") or {}).get("keywords") or {}
            for g in kw.get("generated") or []:
                if isinstance(g.get("level"), int):
                    levels[g["level"]] += 1
                    level_values.append(g["level"])
                elif "kept" in g:  # legacy pdi-v1/v2 records
                    kept_total += 1
                    kept_true += 1 if g["kept"] else 0
            for a in kw.get("added") or []:
                added[a.strip().lower()] += 1
        out = {
            "level_distribution": dict(sorted(levels.items())),
            "level_mean": round(statistics.mean(level_values), 3)
            if level_values
            else None,
            "added_domains_total": sum(added.values()),
            "added_domains_top": added.most_common(15),
        }
        if kept_total:
            out["legacy_kept_rate"] = round(kept_true / kept_total, 3)
        return out

    def pdi_cases(records):
        n_cases = seed_in_top = 0
        seed_ranks, variants, opinions = [], Counter(), []
        for rec in records:
            for case in (rec.get("answers") or {}).get("section5") or []:
                n_cases += 1
                if case.get("seed_in_top"):
                    seed_in_top += 1
                if isinstance(case.get("seed_rank"), int):
                    seed_ranks.append(case["seed_rank"])
                for it in case.get("likert") or []:
                    if it.get("variant"):
                        variants[it["variant"]] += 1
                if (case.get("opinion") or "").strip():
                    opinions.append(case["opinion"].strip())
        return {
            "n_cases": n_cases,
            "seed_in_top_rate": round(seed_in_top / n_cases, 3) if n_cases else None,
            "seed_rank": summary_stats(seed_ranks),
            "position_variants": dict(variants),
            "opinions": opinions,
        }

    def per_researcher(records):
        c = Counter()
        for rec in records:
            rid = rec.get("researcher_id")
            if rid is not None:
                c[str(rid)] += 1
        return {"n_distinct_pdis": len(c), "responses_per_pdi": dict(c)}

    # Student-specific
    def student_context(records):
        schools, levels, used = Counter(), Counter(), Counter()
        favorites = 0
        for rec in records:
            ctx = rec.get("context") or {}
            schools[ctx.get("school") or "unknown"] += 1
            levels[ctx.get("level") or "unknown"] += 1
            used[ctx.get("used_app") or "unknown"] += 1
            if (ctx.get("favorite_topic") or "").strip():
                favorites += 1
        return {
            "by_school": dict(schools),
            "by_level": dict(levels),
            "by_used_app": dict(used),
            "favorite_topic_answered": favorites,
        }

    def student_choices(records):
        multi = Counter()
        single = Counter()
        multi_text = single_text = None
        for rec in records:
            sec2 = (rec.get("answers") or {}).get("section2") or {}
            m = sec2.get("multi") or {}
            multi_text = m.get("text") or multi_text
            labels = {o["value"]: o["label"] for o in m.get("options") or []}
            for v in m.get("selected") or []:
                multi[labels.get(v, v)] += 1
            s = sec2.get("single") or {}
            single_text = s.get("text") or single_text
            slabels = {o["value"]: o["label"] for o in s.get("options") or []}
            if s.get("value") is not None:
                single[slabels.get(s["value"], s["value"])] += 1
        return {
            "multi": {"text": multi_text, "counts": dict(multi.most_common())},
            "single": {"text": single_text, "counts": dict(single.most_common())},
        }

    # Markdown rendering
    def md_meta(meta):
        lines = [
            f"- **Respuestas:** {meta['n_responses']}",
            f"- **Versiones:** {meta['by_form_version']}",
            f"- **Dispositivo:** {meta['by_device']}",
            f"- **Completitud media** — Likert: {meta['mean_likert_completion']} · "
            f"Abiertas: {meta['mean_open_completion']}",
        ]
        d = meta["duration_seconds"]
        if d["n"]:
            lines.append(
                f"- **Duración (s):** media {d['mean']} · mediana {d['median']} (n={d['n']})"
            )
        if meta["by_date"]:
            lines.append(f"- **Por fecha:** {meta['by_date']}")
        return "\n".join(lines)

    def md_likert(table):
        if not table:
            return "_Sin ítems Likert._"
        rows = ["| Ítem | n | Media | SD | Distribución |", "|---|---|---|---|---|"]
        for key, e in table.items():
            note = ""
            if e.get("polarity") == "negative":
                note = f" _(inverso: {e.get('mean_reversed')})_"
            text = (e.get("text") or key).replace("|", "/")
            rows.append(
                f"| {text}{note} | {e['n']} | {e['mean']} | {e['sd']} | {e['distribution']} |"
            )
        return "\n".join(rows)

    def md_open(table):
        if not table:
            return ""
        parts = []
        for key, e in table.items():
            parts.append(f"**{e.get('text') or key}** ({e['answered']} respuestas)")
            for a in e["answers"]:
                parts.append(f"> {a}")
            parts.append("")
        return "\n".join(parts)

    def analyze_pdi(records):
        result = {
            "role": "pdi",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta_quality(records),
            "likert": likert_table(records),
            "keywords": pdi_keywords(records),
            "recommendation_cases": pdi_cases(records),
            "per_researcher": per_researcher(records),
            "open": open_table(records),
        }
        md = [
            "# UPM Match · Análisis del cuestionario de PDIs\n",
            "## Resumen\n",
            md_meta(result["meta"]),
            f"\n- **PDIs distintos:** {result['per_researcher']['n_distinct_pdis']}",
            "\n## Escalas Likert\n",
            md_likert(result["likert"]),
            "\n## Relevancia de dominios de conocimiento\n",
            f"- Distribución de niveles (0-3): {result['keywords']['level_distribution']}",
            f"\n- Nivel medio: {result['keywords']['level_mean']}",
            f"\n- Dominios añadidos: {result['keywords']['added_domains_total']} "
            f"(top: {result['keywords']['added_domains_top']})",
            "\n## Evaluación de casos de recomendación\n",
            f"- Casos evaluados: {result['recommendation_cases']['n_cases']}",
            f"\n- Tasa 'PDI en top-5': {result['recommendation_cases']['seed_in_top_rate']}",
            f"\n- Rank del PDI: {result['recommendation_cases']['seed_rank']}",
            f"\n- Variantes de posición: {result['recommendation_cases']['position_variants']}",
            "\n## Preguntas abiertas\n",
            md_open(result["open"]),
        ]
        return result, "\n".join(md)

    def analyze_student(records):
        result = {
            "role": "student",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta_quality(records),
            "context": student_context(records),
            "choices": student_choices(records),
            "likert": likert_table(records),
            "open": open_table(records),
        }
        md = [
            "# UPM Match · Análisis del cuestionario de estudiantes\n",
            "## Resumen\n",
            md_meta(result["meta"]),
            "\n## Contexto académico\n",
            f"- Escuela: {result['context']['by_school']}",
            f"\n- Nivel: {result['context']['by_level']}",
            f"\n- Uso previo de la app: {result['context']['by_used_app']}",
            "\n## Información más útil (multiselección)\n",
            f"{result['choices']['multi']['counts']}",
            "\n## Criterio a priorizar\n",
            f"{result['choices']['single']['counts']}",
            "\n## Escalas Likert\n",
            md_likert(result["likert"]),
            "\n## Preguntas abiertas\n",
            md_open(result["open"]),
        ]
        return result, "\n".join(md)

    # Stage dispatch
    if stage == "download":
        pdi = load_responses("pdi")
        student = load_responses("student")
        write_json(f"{analysis_prefix}/raw/pdi.json", pdi)
        write_json(f"{analysis_prefix}/raw/student.json", student)
        manifest = {
            "run_ts": run_ts,
            "since": since or None,
            "counts": {"pdi": len(pdi), "student": len(student)},
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(f"{analysis_prefix}/raw/manifest.json", manifest)
        return manifest

    if stage in ("pdi", "student"):
        records = read_json(f"{analysis_prefix}/raw/{stage}.json")
        result, md = (analyze_pdi if stage == "pdi" else analyze_student)(records)
        write_json(f"{analysis_prefix}/{stage}_analysis.json", result)
        write_text(f"{analysis_prefix}/{stage}_report.md", md)
        write_json(f"analysis/latest/{stage}_analysis.json", result)
        write_text(f"analysis/latest/{stage}_report.md", md)
        return {"stage": stage, "n_responses": result["meta"]["n_responses"]}

    if stage == "combine":
        pdi = read_json(f"{analysis_prefix}/pdi_analysis.json")
        student = read_json(f"{analysis_prefix}/student_analysis.json")
        summary = {
            "run_ts": run_ts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pdi_responses": pdi["meta"]["n_responses"],
            "student_responses": student["meta"]["n_responses"],
        }
        report = (
            f"# UPM Match · Informe de cuestionarios ({run_ts})\n\n"
            f"- Respuestas PDI: **{summary['pdi_responses']}**\n"
            f"- Respuestas estudiantes: **{summary['student_responses']}**\n\n"
            "Consulte `pdi_report.md` y `student_report.md` en la misma carpeta "
            f"(`{analysis_prefix}/`) o en `analysis/latest/`.\n"
        )
        write_json(f"{analysis_prefix}/summary.json", summary)
        write_text(f"{analysis_prefix}/report.md", report)
        write_json("analysis/latest/summary.json", summary)
        write_text("analysis/latest/report.md", report)
        print(report)
        return summary

    if stage == "fetch":
        # MinIO has no public route, so mirror `analysis/latest/` into the mounted
        # repo checkout (same convention as upm_match_analysis.fetch_to_repo) and
        # echo the Markdown reports into the task log, which is readable from the
        # Airflow UI even when the mount is not writable by the container user.
        dest = "/opt/airflow/project/evaluation/reports/questionnaire"
        written, failed = [], []
        for obj in client.list_objects(bucket, prefix="analysis/latest/", recursive=True):
            name = obj.object_name.split("/")[-1]
            try:
                os.makedirs(dest, exist_ok=True)
                client.fget_object(bucket, obj.object_name, f"{dest}/{name}")
                written.append(name)
            except Exception as exc:  # noqa: BLE001 — read-only mount must not fail the run
                failed.append(f"{name}: {exc}")
        rule = "=" * 78
        for name in ("pdi_report.md", "student_report.md"):
            try:
                body = read_text(f"analysis/latest/{name}")
            except Exception as exc:  # noqa: BLE001 - the report may not exist yet
                print(f"[{name}] not available: {exc}")
                continue
            print(f"{rule}{chr(10)}{name}{chr(10)}{rule}{chr(10)}{body}")
        if failed:
            print(f"Could not write into {dest} (chmod a+rwx it to keep a copy): {failed}")
        return {"dest": dest, "written": written, "failed": failed}

    raise ValueError(f"Unknown stage: {stage!r}")


_default_args = {
    "owner": "upm-match",
    "retries": 1,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="upm_match_questionnaire",
    description="Download and analyse the PDI + student evaluation questionnaires from MinIO.",
    start_date=datetime(2026, 1, 30),
    schedule=None,  # manual trigger; set a cron (e.g. "0 6 * * 1") for a weekly run.
    catchup=False,
    max_active_runs=1,
    default_args=_default_args,
    params={
        "since": Param(
            "",
            type="string",
            description="Optional YYYY-MM-DD. Only analyse responses submitted on/after "
            "this date (empty = all).",
        ),
    },
    tags=["upm", "questionnaire", "analysis"],
    doc_md=__doc__,
) as dag:
    _common = {
        "python": _VENV_PYTHON,
        "python_callable": run_stage,
        "expect_airflow": False,  # the project venv has no Airflow, only project deps.
    }
    _ts = "{{ ts_nodash }}"
    _since = "{{ params.since }}"

    download = ExternalPythonOperator(
        task_id="download_responses",
        op_kwargs={"stage": "download", "run_ts": _ts, "since": _since},
        doc_md="Pull every response JSON from MinIO, split by role and stage the raw sets.",
        **_common,
    )
    analyze_pdi = ExternalPythonOperator(
        task_id="analyze_pdi",
        op_kwargs={"stage": "pdi", "run_ts": _ts, "since": _since},
        doc_md="Aggregate the PDI responses and publish pdi_analysis.json + pdi_report.md.",
        **_common,
    )
    analyze_students = ExternalPythonOperator(
        task_id="analyze_students",
        op_kwargs={"stage": "student", "run_ts": _ts, "since": _since},
        doc_md="Aggregate the student responses and publish student_analysis.json + report.",
        **_common,
    )
    combine = ExternalPythonOperator(
        task_id="combine_report",
        op_kwargs={"stage": "combine", "run_ts": _ts, "since": _since},
        doc_md="Merge both analyses into report.md + summary.json and log an overview.",
        **_common,
    )

    fetch_to_repo = ExternalPythonOperator(
        task_id="fetch_to_repo",
        op_kwargs={"stage": "fetch", "run_ts": _ts, "since": _since},
        # Best effort, and pulls whatever exists even if a step above failed.
        trigger_rule="all_done",
        doc_md=(
            "Mirrors `analysis/latest/` into the mounted repo checkout "
            "(`evaluation/reports/questionnaire/`) and prints both Markdown "
            "reports into this task log, since MinIO has no public route. "
            "The directory must be writable by the container user "
            "(`mkdir -p evaluation/reports/questionnaire && chmod -R a+rwx "
            "evaluation/reports`); if it is not, the log copy still works."
        ),
        **_common,
    )

    # PDI and student analyses are independent, so they run in parallel after the download and join into the combined report.
    download >> [analyze_pdi, analyze_students] >> combine >> fetch_to_repo
