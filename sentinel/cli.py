"""Interface de linha de comando (seção 15 do CLAUDE.md)."""

import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from sentinel.agent.analyst import write_report
from sentinel.agent.render import render_report
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
from sentinel.memory.factory import open_memory
from sentinel.memory.knowledge import index_files
from sentinel.security import configure_logging, describe_error, redact

app = typer.Typer(
    help="DataSentinel: agente que investiga anomalias em métricas de negócio.",
    no_args_is_help=True,
    # Nunca imprimir variáveis locais (poderiam trazer DSN ou token); `run` trata os erros.
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
)

DEFAULT_SCENARIOS = Path("scenarios/scenarios.jsonl")
DEFAULT_DOCS = [Path("README.md"), Path("SECURITY.md"), Path("docs/DECISIONS.md")]

memory_app = typer.Typer(
    help="Memória de incidentes e busca de contexto (pgvector).", no_args_is_help=True
)
app.add_typer(memory_app, name="memory")


@app.callback()
def main() -> None:
    """DataSentinel."""
    configure_logging(get_settings())


def run() -> None:
    """Ponto de entrada do comando `sentinel`: erros inesperados saem sem dados sensíveis.

    Com `SENTINEL_DEBUG=1` mostra o traceback (sem variáveis locais e com segredos redigidos).
    Mesmo assim, não cole esse traceback em issues públicas sem ler antes.
    """
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("Interrompido.", err=True)
        raise SystemExit(130) from None
    except Exception as error:
        typer.echo(f"Erro: {describe_error(error)}", err=True)
        if os.environ.get("SENTINEL_DEBUG") == "1":
            typer.echo(redact("".join(traceback.format_exception(error))), err=True)
        else:
            typer.echo("Defina SENTINEL_DEBUG=1 para ver o traceback.", err=True)
        raise SystemExit(1) from None


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
        raise typer.BadParameter(redact(f"{db_path} não existe; rode `sentinel generate` antes"))

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
    """Detector + investigação determinística e relatório de template, sem LLM nem Telegram."""
    settings = get_settings()
    db_path = settings.db_path if db_path is None else db_path
    if not db_path.exists():
        raise typer.BadParameter(redact(f"{db_path} não existe; rode `sentinel generate` antes"))

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

    outcome = write_report(None, root, hypotheses, ctx.findings)
    typer.echo(
        f"Anomalia: {root.metric} ({root.direction}), {root.window_start} a {root.window_end}, "
        f"início {root.onset}, z={root.z_score:+.1f}"
    )
    typer.echo(render_report(outcome.report, ctx.findings))
    typer.echo(f"\nEvidências: {', '.join(outcome.report.evidence)}")


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


@memory_app.command("check")
def memory_check() -> None:
    """Confere a conexão com o Postgres e a extensão pgvector (nunca mostra o DSN)."""
    settings = get_settings()
    if not settings.memory_dsn.get_secret_value():
        typer.echo("MEMORY_DSN vazio: a memória fica só em RAM e não persiste.")
        return
    memory = open_memory(settings)
    try:
        assert memory.pg is not None
        for name, value in memory.pg.status().items():
            typer.echo(f"  {name}: {redact(str(value))}")
        typer.echo(f"  trechos de contexto: {memory.knowledge.count()}")
    finally:
        memory.close()


@memory_app.command("index")
def memory_index(
    paths: Annotated[
        list[Path] | None, typer.Argument(help="arquivos .md/.txt; padrão: README e docs")
    ] = None,
) -> None:
    """Indexa documentos na busca de contexto (trechos com segredo são recusados)."""
    memory = open_memory(get_settings())
    try:
        report = index_files(memory.knowledge, paths or DEFAULT_DOCS, root=Path.cwd())
    finally:
        memory.close()
    typer.echo(
        f"[{memory.backend}] novos: {report.added} | mantidos: {report.kept} "
        f"| removidos: {report.removed}"
    )
    for item in report.skipped:
        typer.echo(f"  ignorado: {item}")


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="o que procurar")],
    k: Annotated[int | None, typer.Option(help="quantos resultados")] = None,
) -> None:
    """Busca os trechos de contexto mais parecidos com a pergunta."""
    settings = get_settings()
    memory = open_memory(settings)
    try:
        hits = memory.knowledge.search(query, k or settings.memory_top_k, 0.0)
    finally:
        memory.close()
    if not hits:
        typer.echo("Nada encontrado (rode `sentinel memory index` primeiro).")
    for hit in hits:
        typer.echo(f"{hit.similarity:.2f}  {hit.chunk.label}")
