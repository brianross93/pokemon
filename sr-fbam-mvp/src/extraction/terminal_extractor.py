"""
Terminal entity extraction utilities for SR-FBAM terminal dataset v2.

Provides lightweight pattern matching that converts raw terminal output into
structured entities (files, commands, errors, env vars, processes) along with
coarse relationships between the discovered entities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FILE_PATTERN = re.compile(
    r"(?<![\w/.-])(?P<path>[\w./-]+\.(?:py|pyi|js|ts|tsx|jsx|go|rb|rs|c|cc|cpp|h|hpp|java|cs|php|swift|kt|json|ya?ml|toml|ini|cfg|txt|md|rst|sh|ps1|bat|dockerfile|lock|cfg|conf))",
    re.IGNORECASE,
)
DOCKERFILE_PATTERN = re.compile(r"\bDockerfile(?:\.[\w.-]+)?\b")
COMMAND_LINE_PATTERN = re.compile(r"^[\s]*[>$]\s*(?P<cmd>.+)$")
RUN_LINE_PATTERN = re.compile(r"^\s*(?:##\[[^\]]+\])?\s*Run\s+(?P<cmd>.+)$")
ENV_PATTERN = re.compile(r"\b([A-Z_][A-Z0-9_]*)=([^\s\"']+|\"[^\"]*\"|'[^']*')\b")
PROCESS_PATTERN = re.compile(
    r"(?:(?P<name>[A-Za-z0-9_.-]+)\s*\(pid[:=]\s*(?P<pid>\d+)\)|PID[:=]\s*(?P<pid_only>\d+))",
    re.IGNORECASE,
)
PYTEST_FAIL_PATTERN = re.compile(
    r"FAILED\s+(?P<file>[\w./-]+)::(?P<test>[\w\[\].-]+)(?:\s+-\s+(?P<reason>.+))?",
    re.IGNORECASE,
)
ERROR_PATTERN = re.compile(
    r"(?P<type>(?:\w+)?Error|Exception|FAILED)(?::\s*(?P<message>.+))?",
)
IMPORT_ERROR_PATTERN = re.compile(
    r"ImportError:\s*(?:cannot\s+import\s+name\s+['\"]?(?P<symbol>[\w.]+)['\"]?)",
    re.IGNORECASE,
)
DEF_SYMBOL_PATTERN = re.compile(
    r"(?P<file>[\w./-]+\.(?:py|js|ts|rb|go|rs|java|c|cpp|hpp)):(?:\s*(?:def|function|class)\s+)(?P<symbol>[\w.]+)",
    re.IGNORECASE,
)


@dataclass
class ErrorEntity:
    """Structured representation of an error occurrence in the terminal output."""

    type: str
    message: Optional[str] = None
    file: Optional[str] = None
    test: Optional[str] = None
    reason: Optional[str] = None
    raw: Optional[str] = None
    symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "type": self.type,
            "message": self.message,
            "file": self.file,
            "test": self.test,
            "reason": self.reason,
            "raw": self.raw,
            "symbol": self.symbol,
        }


@dataclass
class ProcessEntity:
    """Representation of a process mention (name, pid, status)."""

    name: Optional[str]
    pid: int
    status: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        payload = {"pid": self.pid}
        if self.name:
            payload["name"] = self.name
        if self.status:
            payload["status"] = self.status
        return payload


@dataclass
class TerminalEntityExtraction:
    entities: Dict[str, object]
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)
    coverage: float = 0.0


class TerminalEntityExtractor:
    """
    Lightweight rule-based entity extractor tuned for terminal debugging logs.

    The extractor favors predictable, interpretable regular expressions so that
    downstream symbolic operators can reuse the structured entities without
    re-encoding the raw transcript.
    """

    def extract(
        self,
        text: str,
        command_hint: Optional[str] = None,
    ) -> TerminalEntityExtraction:
        files = self._extract_files(text)
        commands = self._extract_commands(text, command_hint)
        errors = self._extract_errors(text)
        env_vars = self._extract_env_vars(text)
        processes = self._extract_processes(text)
        relationships = self._build_relationships(commands, files, errors, text)

        entities = {
            "files": sorted(files),
            "commands": sorted(commands),
            "errors": [err.to_dict() for err in errors],
            "env_vars": env_vars,
            "processes": [proc.to_dict() for proc in processes],
        }
        coverage = self.compute_coverage(text=text, entities=entities)
        return TerminalEntityExtraction(entities=entities, relationships=relationships, coverage=coverage)

    def _extract_files(self, text: str) -> List[str]:
        found = {match.group("path") for match in FILE_PATTERN.finditer(text)}
        if DOCKERFILE_PATTERN.search(text):
            for match in DOCKERFILE_PATTERN.finditer(text):
                found.add(match.group(0))
        return list(found)

    def _extract_commands(self, text: str, command_hint: Optional[str]) -> List[str]:
        commands = set()
        if command_hint:
            commands.add(command_hint.strip())
        for match in RUN_LINE_PATTERN.finditer(text):
            cmd = match.group("cmd").strip()
            if cmd:
                commands.add(cmd)
        for match in COMMAND_LINE_PATTERN.finditer(text):
            cmd = match.group("cmd").strip()
            if cmd:
                commands.add(cmd)
        return list(commands)

    def _extract_errors(self, text: str) -> List[ErrorEntity]:
        errors: List[ErrorEntity] = []
        for match in PYTEST_FAIL_PATTERN.finditer(text):
            errors.append(
                ErrorEntity(
                    type="FAILED",
                    message=match.group("reason"),
                    file=match.group("file"),
                    test=match.group("test"),
                    reason=match.group("reason"),
                    raw=match.group(0),
                )
            )
        for match in ERROR_PATTERN.finditer(text):
            err_type = match.group("type")
            message = match.group("message")
            errors.append(ErrorEntity(type=err_type, message=message, raw=match.group(0)))
        for match in IMPORT_ERROR_PATTERN.finditer(text):
            symbol = match.group("symbol")
            errors.append(ErrorEntity(type="ImportError", symbol=symbol, message=match.group(0), raw=match.group(0)))
        return self._deduplicate_errors(errors)

    def _deduplicate_errors(self, errors: Sequence[ErrorEntity]) -> List[ErrorEntity]:
        unique: Dict[Tuple[Optional[str], Optional[str], Optional[str]], ErrorEntity] = {}
        for err in errors:
            key = (err.type, err.message, err.file or err.test or err.symbol)
            if key not in unique:
                unique[key] = err
        return list(unique.values())

    def _extract_env_vars(self, text: str) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for key, value in ENV_PATTERN.findall(text):
            clean = value.strip("\"'")
            env[key] = clean
        return env

    def _extract_processes(self, text: str) -> List[ProcessEntity]:
        processes: List[ProcessEntity] = []
        seen: set[Tuple[Optional[str], int]] = set()
        for match in PROCESS_PATTERN.finditer(text):
            name = match.group("name")
            pid_str = match.group("pid") or match.group("pid_only")
            if not pid_str:
                continue
            pid = int(pid_str)
            status = self._infer_process_status(text_line=match.group(0))
            key = (name, pid)
            if key in seen:
                continue
            seen.add(key)
            processes.append(ProcessEntity(name=name, pid=pid, status=status))
        return processes

    def _infer_process_status(self, text_line: str) -> Optional[str]:
        lowered = text_line.lower()
        for marker in ("running", "stopped", "terminated", "exited", "killed"):
            if marker in lowered:
                return marker
        return None

    def _build_relationships(
        self,
        commands: Iterable[str],
        files: Iterable[str],
        errors: Sequence[ErrorEntity],
        text: str,
    ) -> List[Tuple[str, str, str]]:
        relationships: List[Tuple[str, str, str]] = []
        command_set = list({cmd for cmd in commands})
        file_set = list({f for f in files})
        # command -> file relationships
        for cmd in command_set:
            cmd_lower = cmd.lower()
            verb = None
            if "pytest" in cmd_lower:
                verb = "reads"
            elif any(token in cmd_lower for token in ("grep", "rg", "findstr")):
                verb = "searches"
            elif any(token in cmd_lower for token in ("cat", "less", "tail")):
                verb = "views"
            elif any(token in cmd_lower for token in ("vim", "nano", "code", "sed")):
                verb = "edits"
            elif any(token in cmd_lower for token in ("docker", "compose")):
                verb = "manages"
            if verb:
                for path in file_set:
                    relationships.append((cmd.split()[0], verb, path))

        # error relationships
        for err in errors:
            if err.test and err.file:
                relationships.append((err.test, "failed_in", err.file))
            if err.test and err.symbol:
                relationships.append((err.test, "imports", err.symbol))
            if err.file and err.symbol:
                relationships.append((err.symbol, "defined_in?", err.file))
            if err.reason:
                relationships.append((err.test or err.type, "failure_reason", err.reason))

        # definition relationships gleaned from grep output
        for match in DEF_SYMBOL_PATTERN.finditer(text):
            file_path = match.group("file")
            symbol = match.group("symbol")
            relationships.append((symbol, "defined_in", file_path))

        deduped = []
        seen_rel = set()
        for rel in relationships:
            if rel not in seen_rel:
                seen_rel.add(rel)
                deduped.append(rel)
        return deduped

    def compute_coverage(self, text: str, entities: Dict[str, object]) -> float:
        if not text:
            return 0.0
        total_tokens = max(len(text.split()), 1)
        entity_tokens = 0
        for key, value in entities.items():
            if key == "env_vars" and isinstance(value, dict):
                entity_tokens += sum(len(k.split()) + len(v.split()) for k, v in value.items())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        entity_tokens += len(item.split())
                    elif isinstance(item, dict):
                        entity_tokens += sum(len(str(v).split()) for v in item.values() if v)
        coverage = min(entity_tokens / total_tokens, 1.0)
        return round(coverage, 4)


__all__ = ["TerminalEntityExtractor", "TerminalEntityExtraction"]

