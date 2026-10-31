"""Interface de linha de comando (seção 15 do CLAUDE.md)."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from sentinel.config import get_settings
from sentinel.data.db import connect_readonly
from sentinel.data.generator import (
    generate_events,
    load_scenarios,
    write_database,
    write_ground_truth,
)
from sentinel.detection.detector import DetectorParams, load_daily_series, select_root
from sentinel.detection.detector import detect as detect_anomalies
from sentinel.investigation.plan import investigate as investigate_plan
from sentinel.investigation.tools import ToolContext

app = typer.Typer(
    help="DataSentinel: agente que investiga anomalias em métricas de negócio.",
    no_args_is_help=True,
)

DEFAULT_SCENARIOS = Path("scenarios/scenarios.jsonl")


@app.callback()
def main() -> None:
    """DataSentinel."""


@app.command()
def detect(
    db_path: Annotated[
        Path | None, typer.Option(help="banco de entrada; padrão: DB_PATH do .env")
    ] = None,
    as_of: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="fim da janela; padrão: último dia"),
    ] = None,
) -> None:
    """Roda só o detector sobre o banco (somente leitura)."""
    settings = get_settings()
    db_path = settings.db_path if db_path is None else db_path
    if not db_path.exists():
        raise typer.BadParameter(f"{db_path} não existe; rode `sentinel generate` antes")

    con = connect_readonly(db_path)
    try:
        series = load_daily_series(con)
    finally:
        con.close()

    found = detect_anomalies(
        series, DetectorParams.from_settings(settings), as_of.date() if as_of else None
    )
    if not found:
        typer.echo("Nenhuma anomalia na janela.")
        return
    root = select_root(found)
    for a in found:
        mark = " <- raiz" if a is root else ""
        typer.echo(
            f"{a.metric:16s} {a.direction:5s} z={a.z_score:+.1f}  "
            f"{a.window_start} a {a.window_end}  início {a.onset}  "
            f"observado {a.observed:,.2f}  esperado {a.expected:,.2f}{mark}"
        )


@app.command()
def investigate(
    db_path: Annotated[
        Path | None, typer.Option(help="banco de entrada; padrão: DB_PATH do .env")
    ] = None,
    as_of: Annotated[
        datetime | None,
        typer.Option(formats=["%Y-%m-%d"], help="fim da janela; padrão: último dia"),
    ] = None,
) -> None:
    """Detector + investigação determinística, sem LLM nem Telegram."""
    settings = get_settings()
    db_path = settings.db_path if db_path is None else db_path
    if not db_path.exists():
        raise typer.BadParameter(f"{db_path} não existe; rode `sentinel generate` antes")

    con = connect_readonly(db_path)
    try:
        found = detect_anomalies(
            load_daily_series(con),
            DetectorParams.from_settings(settings),
            as_of.date() if as_of else None,
        )
        root = select_root(found)
        if root is None:
            typer.echo("Nenhuma anomalia na janela.")
            return
        ctx = ToolContext(con, settings)
        hypotheses = investigate_plan(ctx, root)
    finally:
        con.close()

    typer.echo(
        f"Anomalia: {root.metric} ({root.direction}), {root.window_start} a {root.window_end}, "
        f"início {root.onset}, z={root.z_score:+.1f}"
    )
    if not hypotheses:
        typer.echo("Causa não identificada (inconclusivo).")
    for number, h in enumerate(hypotheses, start=1):
        where = ", ".join(f"{k}={v}" for k, v in h.dimensions.items())
        typer.echo(
            f"Causa {number}: {where} (explica {h.explained_share:.0%}) evidências {h.finding_ids}"
        )
    for finding in ctx.findings:
        if finding.tool == "check_recent_changes":
            for change in finding.result["changes"]:
                typer.echo(
                    f"  {finding.id}: {change['ts']} {change['type']}: {change['description']}"
                )


@app.command()
def generate(
    scenario: Annotated[str, typer.Option("--scenario", "-s", help="id do cenário, ex.: s01")],
    seed: Annotated[int | None, typer.Option(help="semente; padrão: SEED do .env")] = None,
    scenarios_file: Annotated[Path, typer.Option(help="arquivo de cenários")] = DEFAULT_SCENARIOS,
    db_path: Annotated[
        Path | None, typer.Option(help="banco de saída; padrão: DB_PATH do .env")
    ] = None,
) -> None:
    """Gera os dados sintéticos com a causa do cenário injetada."""
    settings = get_settings()
    seed = settings.seed if seed is None else seed
    db_path = settings.db_path if db_path is None else db_path

    by_id = {s.id: s for s in load_scenarios(scenarios_file)}
    if scenario not in by_id:
        raise typer.BadParameter(
            f"cenário {scenario!r} não existe; opções: {', '.join(by_id)}", param_hint="--scenario"
        )
    chosen = by_id[scenario]

    write_database(chosen, seed, db_path)
    truth_path = db_path.with_name("ground_truth.json")
    write_ground_truth(chosen, seed, truth_path)

    rows = len(generate_events(chosen, seed))
    typer.echo(f"{chosen.id} ({chosen.kind}): {chosen.description}")
    typer.echo(f"  {rows} linhas em {db_path} | gabarito em {truth_path} | semente {seed}")
