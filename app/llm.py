"""Claude wrapper. Kept tiny so it is easy to swap or fake in tests."""
from . import config


class LLMError(Exception):
    """Raised when the language model call fails."""


class ClaudeLLM:
    name = "claude"

    def __init__(self, api_key: str, model: str, max_tokens: int):
        from anthropic import Anthropic          # imported here so the app runs without the SDK
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc
        return "".join(block.text for block in response.content if block.type == "text").strip()


def build_llm():
    """Return a ClaudeLLM when a key is configured, otherwise None (retrieval-only mode)."""
    if not config.ANTHROPIC_API_KEY:
        return None
    return ClaudeLLM(config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL, config.MAX_TOKENS)
