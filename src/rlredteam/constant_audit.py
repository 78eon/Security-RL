"""Repository-wide symbolic-constant classifier used by the knowledge audit."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConstantFinding:
    path: str
    line: int
    name: str
    category: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


_DEPLOYMENT_NAMES = {
    "REPO_ROOT",
    "RUNS_DIR",
    "RESULTS_DIR",
    "SCHEMA_PATH",
    "CATALOGUE_DB",
    "PROVENANCE_DIR",
    "NVD_ENDPOINT",
}
_MODEL_FILES = {
    "src/rlredteam/enterprise/model.py",
    "src/rlredteam/cvss.py",
    "gui/theme.py",
    "gui/widgets/enterprise_graph.py",
}
_MODEL_NAMES = {"ON_PREM_TYPES"}
_SIMULATED_NAMES = {
    "_CATALOGUE",
    "_REFERENCE",
    "ATTACK_TECHNIQUES",
    "ATLAS_TECHNIQUES",
    "_BY_ID",
    "_ACTION_TECHNIQUE",
    "_AI_TECHNIQUE",
    "KIND_TO_TECHNIQUE",
    "DEMO_PATH",
    "TARGETS",
}


def _category(path: str, name: str, owner: str | None) -> str:
    if name in _SIMULATED_NAMES:
        return "simulated_world"
    if name in _DEPLOYMENT_NAMES or name.endswith("_DIR") or (
        name.startswith("DEFAULT_")
        and any(token in name for token in ("CONFIG", "MANIFEST", "PATH", "ROOT", "DB", "INPUT"))
    ):
        return "deployment"
    if path in _MODEL_FILES or name in _MODEL_NAMES:
        return "model_schema"
    if owner in {
        "NodeType", "EdgeType", "DeploymentProfile", "HybridFamily", "Framework", "AIBehavior"
    }:
        return "model_schema"
    return "protocol"


def classify_python_constants(repo_root: Path) -> list[ConstantFinding]:
    """Classify every uppercase module assignment and enum member deterministically."""
    findings: list[ConstantFinding] = []
    roots = (repo_root / "src", repo_root / "gui", repo_root / "tools")
    for root in roots:
        for source in sorted(root.rglob("*.py")):
            relative = source.relative_to(repo_root).as_posix()
            tree = ast.parse(source.read_text(), filename=relative)
            for statement in tree.body:
                if isinstance(statement, ast.Assign | ast.AnnAssign):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                    )
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            findings.append(
                                ConstantFinding(
                                    relative,
                                    statement.lineno,
                                    target.id,
                                    _category(relative, target.id, None),
                                )
                            )
                if isinstance(statement, ast.ClassDef):
                    is_enum = any(
                        (isinstance(base, ast.Name) and base.id.endswith("Enum"))
                        or (isinstance(base, ast.Attribute) and base.attr.endswith("Enum"))
                        for base in statement.bases
                    )
                    if not is_enum:
                        continue
                    for child in statement.body:
                        if isinstance(child, ast.Assign):
                            for target in child.targets:
                                if isinstance(target, ast.Name) and target.id.isupper():
                                    findings.append(
                                        ConstantFinding(
                                            relative,
                                            child.lineno,
                                            f"{statement.name}.{target.id}",
                                            _category(relative, target.id, statement.name),
                                        )
                                    )
    return findings
