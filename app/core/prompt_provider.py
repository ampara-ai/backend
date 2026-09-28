from pathlib import Path

_PROMPTS_DIR = Path(__file__).parents[2] / "prompts"


class PromptProvider:
    """
    Loads all system prompts from the prompts/ directory at import time.
    Each prompt is a plain .md file — edit them without touching Python code.
    """

    def __init__(self, prompts_dir: Path = _PROMPTS_DIR) -> None:
        self.relevance_filter = self._load(prompts_dir, "relevance_filter.md")
        self.context_check = self._load(prompts_dir, "context_check.md")
        self.generation = self._load(prompts_dir, "generation.md")

    @staticmethod
    def _load(directory: Path, filename: str) -> str:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
                f"Make sure '{filename}' exists in the prompts/ directory."
            )
        return path.read_text(encoding="utf-8").strip()


prompts = PromptProvider()
