"""命令行入口。

用法：
    python -m auto_reply_eval.cli --mode mock
    python -m auto_reply_eval.cli --mode real --model deepseek-chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from auto_reply_eval.config import LLMMode, Settings
from auto_reply_eval.evaluation.pipeline import EvaluationPipeline
from auto_reply_eval.judge.reply_judge import JudgeResponseError
from auto_reply_eval.metrics.definitions import all_definitions
from auto_reply_eval.models import EvaluationResult
from auto_reply_eval.reporting.console import render_console
from auto_reply_eval.reporting.html import render_html
from auto_reply_eval.reporting.markdown import render_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto_reply_eval",
        description="客服自动回复质量评估流水线",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in LLMMode],
        default=None,
        help="裁判模式：mock 为离线模拟，real 调用真实 LLM",
    )
    parser.add_argument("--model", default=None, help="真实模式下的模型名")
    parser.add_argument("--base-url", default=None, help="真实模式下的 OpenAI 兼容接口地址")
    parser.add_argument("--dataset", type=Path, default=None, help="自动回复数据集路径")
    parser.add_argument("--human-ref", type=Path, default=None, help="人工参考回复路径")
    parser.add_argument("--report-dir", type=Path, default=None, help="报告输出目录")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条，便于快速验证")
    parser.add_argument("--quiet", action="store_true", help="不打印逐条评分进度")
    return parser


def resolve_settings(args: argparse.Namespace) -> Settings:
    """命令行参数优先于 .env 与环境变量。"""

    overrides: dict[str, object] = {}
    if args.mode is not None:
        overrides["mode"] = LLMMode(args.mode)
    if args.model is not None:
        overrides["model"] = args.model
    if args.base_url is not None:
        overrides["base_url"] = args.base_url
    if args.dataset is not None:
        overrides["dataset_path"] = args.dataset
    if args.human_ref is not None:
        overrides["human_ref_path"] = args.human_ref
    if args.report_dir is not None:
        overrides["report_dir"] = args.report_dir
    return Settings(**overrides)


def write_reports(
    result: EvaluationResult,
    report_dir: Path,
    pages_dir: Path | None = None,
) -> list[Path]:
    """把结果写成 Markdown / HTML / JSON 三种产物。

    传入 ``pages_dir`` 时会额外写一份 ``index.html``，
    使 GitHub Pages 直接指向该目录即可得到线上地址，无需手工搬文件。
    """

    report_dir.mkdir(parents=True, exist_ok=True)
    definitions = list(all_definitions())

    markdown_path = report_dir / "eval_report.md"
    markdown_path.write_text(render_markdown(result), encoding="utf-8")

    html_content = render_html(result, definitions)
    html_path = report_dir / "eval_report.html"
    html_path.write_text(html_content, encoding="utf-8")

    json_path = report_dir / "eval_result.json"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    paths = [markdown_path, html_path, json_path]

    if pages_dir is not None:
        pages_dir.mkdir(parents=True, exist_ok=True)
        index_path = pages_dir / "index.html"
        index_path.write_text(html_content, encoding="utf-8")
        paths.append(index_path)

    return paths


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    args = build_parser().parse_args(argv)
    settings = resolve_settings(args)

    progress = None if args.quiet else (lambda message: print(message, flush=True))
    try:
        pipeline = EvaluationPipeline.from_settings(settings)
        result = pipeline.run(limit=args.limit, on_progress=progress)
    except (ValueError, JudgeResponseError) as error:
        print(f"评估失败：{error}", file=sys.stderr)
        return 1

    print()
    print(render_console(result))
    print()
    print("报告产物：")
    for path in write_reports(result, settings.report_dir, settings.pages_dir):
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
