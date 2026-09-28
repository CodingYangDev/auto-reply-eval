"""全局运行配置。

所有可调参数集中在此处，避免散落在业务代码中的硬编码；
支持通过环境变量或 ``.env`` 文件覆盖（前缀 ``EVAL_``）。
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMMode(str, Enum):
    """裁判模型的运行模式。"""

    MOCK = "mock"
    REAL = "real"


class Settings(BaseSettings):
    """评估流水线的配置项。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="EVAL_",
        extra="ignore",
    )

    mode: LLMMode = LLMMode.MOCK

    # 真实 LLM 裁判的接入参数，任何兼容 OpenAI Chat Completions 协议的服务均可
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 90.0
    json_mode: bool = True
    max_retries: int = 2

    # 输入数据与输出目录
    dataset_path: Path = PROJECT_ROOT / "task3_auto_replies.json"
    human_ref_path: Path = PROJECT_ROOT / "task3_human_ref.json"
    report_dir: Path = PROJECT_ROOT / "reports"

    # 评分口径
    worst_case_count: int = 3
    pass_threshold: float = 3.5
