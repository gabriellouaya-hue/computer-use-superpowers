from pathlib import Path
from typing import Any, Literal, cast

from anthropic.types.beta import BetaToolUnionParam

from .base import BaseAnthropicTool, CLIResult, ToolError

Command = Literal["list", "read", "search"]
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class SkillTool(BaseAnthropicTool):
    name: Literal["skill"] = "skill"

    def to_params(self) -> BetaToolUnionParam:
        return cast(
            BetaToolUnionParam,
            {
                "name": self.name,
                "description": (
                    "Use this tool to access reusable skills and procedures before or during a task. "
                    "Skills are local playbooks that describe how to perform common workflows."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["list", "read", "search"],
                            "description": "The skill action to perform.",
                        },
                        "skill_name": {
                            "type": "string",
                            "description": "The name of the skill file, without extension, used by the read command.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query used by the search command.",
                        },
                    },
                    "required": ["command"],
                },
            },
        )

    async def __call__(
        self,
        *,
        command: Command,
        skill_name: str | None = None,
        query: str | None = None,
        **kwargs,
    ):
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        if command == "list":
            skills = sorted(path.stem for path in SKILLS_DIR.glob("*.md"))
            if not skills:
                return CLIResult(output="No skills are available.")
            return CLIResult(output="Available skills:\n" + "\n".join(f"- {skill}" for skill in skills))

        if command == "read":
            if not skill_name:
                raise ToolError("skill_name is required for command=read")
            skill_path = self._resolve_skill_path(skill_name)
            if not skill_path.exists():
                raise ToolError(f"Skill '{skill_name}' was not found")
            return CLIResult(output=skill_path.read_text())

        if command == "search":
            if not query:
                raise ToolError("query is required for command=search")

            matches: list[str] = []
            lowered_query = query.casefold()
            for skill_path in sorted(SKILLS_DIR.glob("*.md")):
                content = skill_path.read_text()
                if lowered_query in skill_path.stem.casefold() or lowered_query in content.casefold():
                    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
                    matches.append(f"- {skill_path.stem}: {first_line}")

            if not matches:
                return CLIResult(output=f"No skills matched query: {query}")
            return CLIResult(output="Matching skills:\n" + "\n".join(matches))

        raise ToolError(f"Unsupported command: {command}")

    def _resolve_skill_path(self, skill_name: str) -> Path:
        normalized_name = Path(skill_name).stem
        if any(separator in skill_name for separator in ("/", "\\")):
            raise ToolError("skill_name must not contain path separators")
        return SKILLS_DIR / f"{normalized_name}.md"
