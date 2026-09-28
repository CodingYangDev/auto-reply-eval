"""裁判客户端工厂：根据配置产出具体实现。"""

from __future__ import annotations

from auto_reply_eval.config import LLMMode, Settings
from auto_reply_eval.llm.base import LLMClient
from auto_reply_eval.llm.cache import CachedLLMClient
from auto_reply_eval.llm.mock_client import MockLLMClient
from auto_reply_eval.llm.openai_client import OpenAICompatibleClient


def create_llm_client(settings: Settings) -> LLMClient:
    """按 ``settings.mode`` 创建裁判客户端。

    只对真实模型套缓存：mock 本身是确定性的、也没有网络开销，缓存对它没有意义。
    """

    if settings.mode is LLMMode.MOCK:
        return MockLLMClient()

    client: LLMClient = OpenAICompatibleClient(settings)
    if not settings.use_cache:
        return client
    cache_file = settings.cache_dir / f"judge_cache_{settings.mode.value}.json"
    return CachedLLMClient(client, cache_file)
