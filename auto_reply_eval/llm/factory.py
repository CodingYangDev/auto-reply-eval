"""裁判客户端工厂：根据配置产出具体实现。"""

from __future__ import annotations

from auto_reply_eval.config import LLMMode, Settings
from auto_reply_eval.llm.base import LLMClient
from auto_reply_eval.llm.cache import CachedLLMClient
from auto_reply_eval.llm.mock_client import MockLLMClient
from auto_reply_eval.llm.openai_client import OpenAICompatibleClient


def create_llm_client(settings: Settings) -> LLMClient:
    """按 ``settings.mode`` 创建裁判客户端，并按需套上磁盘缓存。"""

    client: LLMClient = (
        OpenAICompatibleClient(settings)
        if settings.mode is LLMMode.REAL
        else MockLLMClient()
    )
    if settings.use_cache:
        cache_file = settings.cache_dir / f"judge_cache_{settings.mode.value}.json"
        return CachedLLMClient(client, cache_file)
    return client
