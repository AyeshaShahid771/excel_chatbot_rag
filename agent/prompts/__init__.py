from pathlib import Path

_here = Path(__file__).parent
# Prefer 'prompts.txt' (existing file in repo). Fall back to 'prompt.txt' for backwards compatibility.
prompts_file = _here / "prompts.txt"
if not prompts_file.exists():
    prompts_file = _here / "prompt.txt"

if not prompts_file.exists():
    raise FileNotFoundError(
        f"No prompts file found in { _here }. Expected 'prompts.txt' or 'prompt.txt'."
    )

_text = prompts_file.read_text(encoding="utf-8")


class prompts:
    scout_system_prompt = _text


__all__ = ["prompts"]
