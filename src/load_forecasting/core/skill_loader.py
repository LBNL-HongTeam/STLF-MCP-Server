"""
Skills over MCP: locate, parse and serve the agent skills bundled with this server.

Why this exists
---------------
A skill (``SKILL.md`` with ``name``/``description`` front matter, per the open
Agent Skills format) tells an agent *how to sequence* the MCP tools.  Client
hosts discover skills from client-specific directories (Codex:
``.agents/skills``; Claude Code: ``.claude/skills``), which only helps when the
agent's working directory is this repository.  A user who merely configured
the MCP server -- Claude Desktop, a Codex chat folder, an HTTP deployment --
never sees them.

Serving the skills *through the server* removes that dependency.  This module
is the single source of truth used by:

- the ``list_skills`` / ``get_skill`` MCP tools (work in every client today);
- the ``skill://index.json`` and ``skill://{name}/SKILL.md`` MCP resources
  (the interim shape recommended by the MCP "Skills Over MCP" working group,
  forward-compatible with the proposed ``skills/list`` / ``skills/activate``
  primitives);
- the server ``instructions`` string.

Where skills live
-----------------
Canonical location is the repository's top-level ``skills/<name>/SKILL.md``
(that is where client-native discovery symlinks point).  For wheel installs
the same tree is force-included at ``load_forecasting/skills/`` so the tools
keep working without a source checkout.
"""

from importlib.resources import files
from pathlib import Path
from typing import Optional
import logging
import re

import yaml

from .paths import repo_root

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"

# Supporting files larger than this are refused (skills are prose, not data).
_MAX_FILE_BYTES = 1_000_000

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


class SkillNotFoundError(Exception):
    """Raised when a skill (or one of its files) does not exist."""


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def skills_root() -> Optional[Path]:
    """Return the directory holding ``<name>/SKILL.md`` bundles, or None.

    Prefers the source checkout's ``skills/`` (edits are live), then the copy
    packaged inside the wheel.
    """
    root = repo_root()
    if root is not None and (root / "skills").is_dir():
        return root / "skills"
    try:
        packaged = files("load_forecasting").joinpath("skills")
        packaged_path = Path(str(packaged))
        if packaged_path.is_dir():
            return packaged_path
    except Exception as e:  # pragma: no cover - defensive; importlib quirks
        logger.debug("packaged skills lookup failed: %s", e)
    return None


def _skill_dir(name: str) -> Path:
    """Resolve a skill name to its directory, refusing path tricks."""
    root = skills_root()
    if root is None:
        raise SkillNotFoundError("No skills directory is available on this server.")
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise SkillNotFoundError(f"Invalid skill name: {name!r}")
    skill_dir = root / name
    if not (skill_dir / SKILL_FILENAME).is_file():
        available = ", ".join(s["name"] for s in list_skills()) or "(none)"
        raise SkillNotFoundError(f"Skill not found: {name!r}. Available: {available}")
    return skill_dir


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split ``SKILL.md`` into (front-matter dict, markdown body)."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        logger.warning("Bad skill front matter: %s", e)
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[match.end():]


def top_level_sections(body: str) -> list[str]:
    """Return the ``##`` headings of a skill body, in order (its table of contents)."""
    return [m.group(2) for m in _HEADING_RE.finditer(body) if len(m.group(1)) == 2]


def extract_section(body: str, section: str) -> Optional[str]:
    """Return one heading's subtree (heading line through to the next heading of
    the same or higher level).  ``section`` matches a heading case-insensitively
    as a substring, so ``"Section 3"`` finds ``"## Section 3 — Model decision matrix"``.
    """
    needle = section.strip().lower()
    headings = list(_HEADING_RE.finditer(body))
    for i, m in enumerate(headings):
        if needle not in m.group(2).lower():
            continue
        level = len(m.group(1))
        end = len(body)
        for later in headings[i + 1:]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        return body[m.start():end].rstrip() + "\n"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_skills() -> list[dict]:
    """Lightweight metadata for every bundled skill (mirrors ``skills/list``)."""
    root = skills_root()
    if root is None:
        return []
    out: list[dict] = []
    for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        skill_md = skill_dir / SKILL_FILENAME
        if not skill_md.is_file():
            continue
        text = skill_md.read_text(encoding="utf-8")
        meta, body = parse_front_matter(text)
        name = str(meta.get("name") or skill_dir.name)
        if name != skill_dir.name:
            logger.warning(
                "Skill front matter name %r differs from directory %r; using directory name",
                name, skill_dir.name,
            )
            name = skill_dir.name
        out.append(
            {
                "name": name,
                "description": str(meta.get("description") or "").strip(),
                "sections": top_level_sections(body),
                "size_chars": len(text),
                "supporting_files": supporting_files(skill_dir),
            }
        )
    return out


def supporting_files(skill_dir: Path) -> list[str]:
    """Relative paths of every file in the skill bundle other than SKILL.md."""
    return sorted(
        str(p.relative_to(skill_dir))
        for p in skill_dir.rglob("*")
        if p.is_file() and p.name != SKILL_FILENAME and not p.name.startswith(".")
    )


def get_skill(name: str, section: Optional[str] = None) -> dict:
    """Full skill bundle (mirrors ``skills/activate``), or one section of it.

    Raises:
        SkillNotFoundError: unknown skill, or ``section`` matched no heading.
    """
    skill_dir = _skill_dir(name)
    text = (skill_dir / SKILL_FILENAME).read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)
    result = {
        "name": name,
        "description": str(meta.get("description") or "").strip(),
        "sections": top_level_sections(body),
        "supporting_files": supporting_files(skill_dir),
        "source_path": str(skill_dir / SKILL_FILENAME),
    }
    if section:
        chunk = extract_section(body, section)
        if chunk is None:
            raise SkillNotFoundError(
                f"No heading matching {section!r} in skill {name!r}. "
                f"Top-level sections: {result['sections']}"
            )
        result["section"] = section
        result["content"] = chunk
    else:
        result["content"] = body.lstrip("\n")
    return result


def get_skill_file(name: str, filename: str) -> dict:
    """A supporting file from the skill bundle, as text.

    Raises:
        SkillNotFoundError: unknown skill/file, path escapes the bundle, or the
            file is too large / not text.
    """
    skill_dir = _skill_dir(name).resolve()
    target = (skill_dir / filename).resolve()
    if skill_dir not in target.parents:
        raise SkillNotFoundError(f"Invalid file path: {filename!r}")
    if not target.is_file():
        raise SkillNotFoundError(
            f"File {filename!r} not found in skill {name!r}. "
            f"Available: {supporting_files(skill_dir) or '(none)'}"
        )
    size = target.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise SkillNotFoundError(
            f"File {filename!r} is {size} bytes; skill files above {_MAX_FILE_BYTES} are not served."
        )
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise SkillNotFoundError(f"File {filename!r} is not UTF-8 text.")
    return {
        "name": name,
        "filename": str(target.relative_to(skill_dir)),
        "content": content,
        "size_bytes": size,
    }
