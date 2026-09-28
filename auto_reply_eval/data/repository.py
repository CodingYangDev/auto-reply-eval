"""数据集读取。

把 JSON 文件读取与领域模型解耦，路径由配置注入，方便后续换成数据库或对象存储。
"""

from __future__ import annotations

import json
from pathlib import Path

from auto_reply_eval.models import AutoReplyCase, HumanReference


class DatasetRepository:
    """从本地 JSON 读取待评估数据与人工参考。"""

    def __init__(self, dataset_path: Path, human_ref_path: Path) -> None:
        self._dataset_path = dataset_path
        self._human_ref_path = human_ref_path

    def load_cases(self) -> list[AutoReplyCase]:
        raw = self._read_json_list(self._dataset_path)
        return [AutoReplyCase.model_validate(item) for item in raw]

    def load_human_references(self) -> dict[str, HumanReference]:
        if not self._human_ref_path.exists():
            return {}
        raw = self._read_json_list(self._human_ref_path)
        references = [HumanReference.model_validate(item) for item in raw]
        return {reference.id: reference for reference in references}

    @staticmethod
    def _read_json_list(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"数据文件格式错误，期望 JSON 数组：{path}")
        return payload
