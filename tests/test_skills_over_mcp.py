"""Tests for Skills over MCP: the skill loader, the list_skills/get_skill tools,
and the skill:// resources.

The point of the feature is portability: a client that only configured the MCP
server (no repo checkout, arbitrary cwd) must be able to discover and read the
bundled workflow skill.  Tests therefore cover both the source-checkout and
the packaged (wheel) lookup paths, and drive the resources through a real
FastMCP client.
"""

import asyncio
import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core import skill_loader
from load_forecasting.core.skill_loader import (
    SkillNotFoundError,
    extract_section,
    parse_front_matter,
    top_level_sections,
)
from load_forecasting.tools import get_skill, list_skills


BUNDLED = "train-forecast-model"

SAMPLE_SKILL = """---
name: demo-skill
description: A demo skill for tests.
---

# demo-skill

Intro paragraph.

## Section 1 — Setup

Do the setup.

### 1.1 — Detail

Nested detail.

## Section 2 — Run

Run it.
"""


@pytest.fixture
def skills_tree(tmp_path, monkeypatch):
    """A throwaway skills root with one skill plus a supporting file."""
    root = tmp_path / "skills"
    skill = root / "demo-skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(SAMPLE_SKILL)
    (skill / "references" / "table.md").write_text("| a | b |\n")
    (root / "not-a-skill").mkdir()  # dir without SKILL.md must be ignored
    monkeypatch.setattr(skill_loader, "skills_root", lambda: root)
    return root


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestParsing:
    def test_front_matter_split(self):
        meta, body = parse_front_matter(SAMPLE_SKILL)
        assert meta == {"name": "demo-skill", "description": "A demo skill for tests."}
        assert body.lstrip().startswith("# demo-skill")

    def test_no_front_matter(self):
        meta, body = parse_front_matter("# plain\n")
        assert meta == {} and body == "# plain\n"

    def test_top_level_sections_are_h2_only(self):
        _, body = parse_front_matter(SAMPLE_SKILL)
        assert top_level_sections(body) == ["Section 1 — Setup", "Section 2 — Run"]

    def test_extract_section_includes_nested_headings_and_stops_at_sibling(self):
        _, body = parse_front_matter(SAMPLE_SKILL)
        chunk = extract_section(body, "section 1")
        assert chunk.startswith("## Section 1 — Setup")
        assert "### 1.1 — Detail" in chunk
        assert "Section 2" not in chunk

    def test_extract_section_missing(self):
        _, body = parse_front_matter(SAMPLE_SKILL)
        assert extract_section(body, "Section 9") is None


# ---------------------------------------------------------------------------
# Loader against a synthetic tree
# ---------------------------------------------------------------------------

class TestLoader:
    def test_list_skills(self, skills_tree):
        skills = skill_loader.list_skills()
        assert [s["name"] for s in skills] == ["demo-skill"]
        s = skills[0]
        assert s["description"] == "A demo skill for tests."
        assert s["sections"] == ["Section 1 — Setup", "Section 2 — Run"]
        assert s["supporting_files"] == ["references/table.md"]
        assert s["size_chars"] == len(SAMPLE_SKILL)

    def test_get_skill_full(self, skills_tree):
        s = skill_loader.get_skill("demo-skill")
        assert s["content"].startswith("# demo-skill")
        assert "---" not in s["content"].splitlines()[0]  # front matter stripped
        assert s["source_path"].endswith("demo-skill/SKILL.md")

    def test_get_skill_section(self, skills_tree):
        s = skill_loader.get_skill("demo-skill", section="Run")
        assert s["section"] == "Run"
        assert s["content"].strip() == "## Section 2 — Run\n\nRun it."

    def test_get_skill_file(self, skills_tree):
        f = skill_loader.get_skill_file("demo-skill", "references/table.md")
        assert f["filename"] == "references/table.md"
        assert f["content"] == "| a | b |\n"

    @pytest.mark.parametrize("name", ["missing", "../demo-skill", "demo-skill/..", "", "a\\b"])
    def test_bad_skill_names_rejected(self, skills_tree, name):
        with pytest.raises(SkillNotFoundError):
            skill_loader.get_skill(name)

    @pytest.mark.parametrize("path", ["../SKILL.md", "../../pyproject.toml", "/etc/passwd", "references/../../not-a-skill/x"])
    def test_file_paths_cannot_escape_bundle(self, skills_tree, path):
        with pytest.raises(SkillNotFoundError, match="Invalid file path|not found"):
            skill_loader.get_skill_file("demo-skill", path)

    def test_dotdot_that_resolves_back_inside_bundle_is_allowed(self, skills_tree):
        # The guard is on the *resolved* location, not on the presence of "..".
        f = skill_loader.get_skill_file("demo-skill", "references/../references/table.md")
        assert f["filename"] == "references/table.md"

    def test_oversized_file_refused(self, skills_tree, monkeypatch):
        big = skills_tree / "demo-skill" / "big.txt"
        big.write_bytes(b"x" * 10)
        monkeypatch.setattr(skill_loader, "_MAX_FILE_BYTES", 5)
        with pytest.raises(SkillNotFoundError, match="not served"):
            skill_loader.get_skill_file("demo-skill", "big.txt")

    def test_packaged_fallback_when_no_source_checkout(self, tmp_path, monkeypatch):
        """Wheel installs have no repo root; the loader must use the packaged copy."""
        packaged = tmp_path / "site-packages" / "load_forecasting" / "skills" / "pkg-skill"
        packaged.mkdir(parents=True)
        (packaged / "SKILL.md").write_text("---\nname: pkg-skill\ndescription: packaged\n---\n# pkg\n")

        monkeypatch.setattr(skill_loader, "repo_root", lambda: None)

        class _Trav:
            def joinpath(self, sub):
                return tmp_path / "site-packages" / "load_forecasting" / sub

        monkeypatch.setattr(skill_loader, "files", lambda pkg: _Trav())
        assert skill_loader.skills_root() == tmp_path / "site-packages" / "load_forecasting" / "skills"
        assert [s["name"] for s in skill_loader.list_skills()] == ["pkg-skill"]

    def test_no_skills_root(self, monkeypatch):
        monkeypatch.setattr(skill_loader, "skills_root", lambda: None)
        assert skill_loader.list_skills() == []
        with pytest.raises(SkillNotFoundError, match="No skills directory"):
            skill_loader.get_skill("anything")


# ---------------------------------------------------------------------------
# Tools (response envelope) against the real bundled skill
# ---------------------------------------------------------------------------

class TestTools:
    def test_list_skills_tool(self):
        r = list_skills()
        assert r["success"] is True
        names = {s["name"] for s in r["skills"]}
        assert BUNDLED in names
        assert r["count"] == len(r["skills"])
        assert "get_skill" in r["usage"]

    def test_get_skill_tool_full(self):
        r = get_skill(BUNDLED)
        assert r["success"] is True
        assert r["name"] == BUNDLED
        assert "Section 0" in r["content"]
        assert len(r["sections"]) >= 10

    def test_get_skill_tool_section(self):
        r = get_skill(BUNDLED, section="Tool cheat sheet")
        assert r["success"] is True
        assert r["content"].startswith("## Section 11")
        assert len(r["content"]) < 5000

    def test_get_skill_tool_errors(self):
        assert get_skill("nope")["success"] is False
        assert get_skill(BUNDLED, section="Section 99")["success"] is False
        assert get_skill(BUNDLED, file="../pyproject.toml")["success"] is False
        both = get_skill(BUNDLED, section="x", file="y")
        assert both["success"] is False and "not both" in both["error"]


# ---------------------------------------------------------------------------
# Resources + instructions through a real FastMCP client
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_skill_resources_and_instructions():
    from fastmcp import Client
    from load_forecasting.server import mcp

    async def go():
        async with Client(mcp) as c:
            uris = {str(r.uri) for r in await c.list_resources()}
            templates = {t.uriTemplate for t in await c.list_resource_templates()}
            assert "skill://index.json" in uris
            assert "skill://{name}/SKILL.md" in templates
            assert "skill://{name}/files/{path*}" in templates

            index = json.loads((await c.read_resource("skill://index.json"))[0].text)
            assert BUNDLED in {s["name"] for s in index["skills"]}

            md = (await c.read_resource(f"skill://{BUNDLED}/SKILL.md"))[0].text
            assert md.startswith(f"# {BUNDLED}")

            tools = {t.name for t in await c.list_tools()}
            assert {"list_skills", "get_skill"} <= tools

    _run(go())

    text = mcp.instructions or ""
    assert "list_skills" in text and "get_skill" in text and BUNDLED in text
    assert "skill://index.json" in text


# ---------------------------------------------------------------------------
# The real bundle: two skills, and supporting files served both ways
# ---------------------------------------------------------------------------

EXPLORE = "explore-load-data"
HP_REF = "references/hyperparameter-reference.md"
MERGE_REF = "references/multi-csv-merge-protocol.md"


def test_bundle_has_both_skills_with_descriptions():
    r = list_skills()
    by_name = {s["name"]: s for s in r["skills"]}
    assert {BUNDLED, EXPLORE} <= set(by_name)
    for name in (BUNDLED, EXPLORE):
        assert by_name[name]["description"].startswith("Use when"), name
        assert len(by_name[name]["sections"]) >= 5, name
    assert by_name[BUNDLED]["supporting_files"] == [HP_REF, MERGE_REF]
    assert by_name[EXPLORE]["supporting_files"] == []


def test_training_skill_supporting_files_via_tool():
    hp = get_skill(BUNDLED, file=HP_REF)
    assert hp["success"] is True and hp["filename"] == HP_REF
    assert "XGBoost" in hp["content"] and "n_epochs" in hp["content"]
    merge = get_skill(BUNDLED, file=MERGE_REF)
    assert merge["success"] is True
    assert "2.5.4" in merge["content"]          # subsection numbering preserved for cross-refs
    # The main skill points at both files instead of inlining them.
    body = get_skill(BUNDLED)["content"]
    assert HP_REF in body and MERGE_REF in body
    assert "### 2.5.4" not in body


def test_explore_skill_points_at_data_tools_and_hands_off():
    body = get_skill(EXPLORE)["content"]
    for tool in ("list_datasets", "inspect_data", "generate_data_report"):
        assert f"`{tool}`" in body or f"`{tool}(" in body, tool
    assert BUNDLED in body                      # hand-off to the training skill
    for key in ("future_covariate_candidates", "target_excluded", "categorical_code_columns", "split_strategy"):
        assert key in body, key


def test_supporting_file_resource_matches_tool():
    from fastmcp import Client
    from load_forecasting.server import mcp

    async def go():
        async with Client(mcp) as c:
            res = (await c.read_resource(f"skill://{BUNDLED}/files/{HP_REF}"))[0].text
            index = json.loads((await c.read_resource("skill://index.json"))[0].text)
            return res, {s["name"] for s in index["skills"]}

    res, names = _run(go())
    assert res == get_skill(BUNDLED, file=HP_REF)["content"]
    assert {BUNDLED, EXPLORE} <= names
