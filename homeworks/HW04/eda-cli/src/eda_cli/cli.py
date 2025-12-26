from pathlib import Path
import typer
from typing import Annotated
import pandas as pd
from . import core, viz

app = typer.Typer()


@app.command("overview")
def overview(
    input_path: Annotated[
        str,
        typer.Argument(help="Путь к входному CSV-файлу"),
    ]
):
    print(f"Читаю {input_path}...")
    df = pd.read_csv(input_path)
    summary = core.summarize_dataset(df)
    flat_df = core.flatten_summary_for_print(summary)
    print(flat_df.to_string(index=False))


@app.command("report")
def report(
    input_path: Annotated[
        str,
        typer.Argument(help="Путь к входному CSV-файлу"),
    ],
    out_dir: Annotated[
        str,
        typer.Option("--out-dir", help="Папка для сохранения отчёта"),
    ] = "reports",
    max_hist_columns: Annotated[
        int,
        typer.Option(
            "--max-hist-columns",
            help="Максимальное количество числовых столбцов для гистограмм (по умолчанию 6)",
        ),
    ] = 6,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            help="Заголовок отчёта (по умолчанию 'EDA Report')",
        ),
    ] = "EDA Report",
    min_missing_share: Annotated[
        float,
        typer.Option(
            "--min-missing-share",
            help="Минимальная доля пропусков для включения в отчёт (по умолчанию 0.05)",
        ),
    ] = 0.05,
):
    print(f"Читаю {input_path}...")
    df = pd.read_csv(input_path)

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)
    corr_matrix = core.correlation_matrix(df)
    top_cats = core.top_categories(df)

    quality_flags = core.compute_quality_flags(df, summary, missing_df)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append(f"# {title}")
    report_lines.append("")
    report_lines.append(f"## Информация о датасете")
    report_lines.append(f"- Строк: {summary.n_rows}")
    report_lines.append(f"- Столбцов: {summary.n_cols}")
    report_lines.append("")

    report_lines.append(f"## Качество данных")
    report_lines.append(f"- Оценка качества: {quality_flags['quality_score']:.2f}")
    report_lines.append(f"- Слишком мало строк: {quality_flags['too_few_rows']}")
    report_lines.append(f"- Слишком много столбцов: {quality_flags['too_many_columns']}")
    report_lines.append(f"- Макс. доля пропусков: {quality_flags['max_missing_share']:.2%}")
    report_lines.append(f"- Есть постоянные столбцы: {quality_flags['has_constant_columns']}")
    report_lines.append(f"- Есть столбцы с >90% нулей: {quality_flags['has_many_zero_values']}")
    report_lines.append(f"- Макс. гистограмм: {max_hist_columns}")
    report_lines.append(f"- Мин. доля пропусков для отчёта: {min_missing_share:.2%}")
    report_lines.append("")

    print("Генерирую визуализации...")
    hist_paths = viz.plot_histograms_per_column(df, out_dir_path, max_columns=max_hist_columns)
    missing_path = viz.plot_missing_matrix(df, out_dir_path / "missing_matrix.png")
    corr_path = viz.plot_correlation_heatmap(df, out_dir_path / "correlation_heatmap.png")
    top_cat_paths = viz.save_top_categories_tables(top_cats, out_dir_path)

    report_lines.append(f"## Визуализации")
    report_lines.append(f"![](./{hist_paths[0].name})" if hist_paths else "Нет гистограмм.")
    report_lines.append(f"![](./{missing_path.name})")
    report_lines.append(f"![](./{corr_path.name})")
    report_lines.append("")

    report_content = "\n".join(report_lines)
    report_path = out_dir_path / "report.md"
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Отчёт сохранён в {report_path}")

    print("Готово!")


if __name__ == "__main__":
    app()