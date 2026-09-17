"""
Skills-over-MCP tools.

Serve the agent skills bundled with this server (``skills/<name>/SKILL.md``)
so that any MCP client -- Claude Desktop, Claude Code, Codex, a custom agent
-- can retrieve the recommended workflow without the skill being installed
on the client machine.  See ``core/skill_loader.py`` for the rationale and
the matching ``skill://`` resources registered in ``server.py``.
"""

from typing import Optional
import logging

from ..core.skill_loader import (
    SkillNotFoundError,
    get_skill as _get_skill,
    get_skill_file as _get_skill_file,
    list_skills as _list_skills,
)
from ._common import create_success_response, create_error_response

logger = logging.getLogger(__name__)


def list_skills() -> dict:
    """
    List the workflow skills bundled with this server.

    IMPORTANT: call this before starting any multi-step forecasting task
    (training, tuning, evaluating, backtesting, merging weather, building
    reports). Each skill is a step-by-step guide that sequences the MCP tools
    correctly and names the checks an agent must confirm with the user first.
    Follow with get_skill(name) to load the instructions.

    Returns:
        Dict with ``skills`` (name, description, top-level ``sections``,
        ``supporting_files``) and ``count``.
    """
    try:
        skills = _list_skills()
        return create_success_response(
            skills=skills,
            count=len(skills),
            usage="Call get_skill(name) for the full instructions, or "
                  "get_skill(name, section=...) for one section of a long skill.",
        )
    except Exception as e:
        logger.exception("list_skills failed")
        return create_error_response(f"Failed to list skills: {e}")


def get_skill(
    name: str,
    section: Optional[str] = None,
    file: Optional[str] = None,
) -> dict:
    """
    Load a bundled skill's instructions (or one section, or a supporting file).

    Call before starting the workflow the skill covers, then follow it step by
    step. The skill is authoritative on tool ordering and on which
    assumptions must be confirmed with the user before spending compute.

    Args:
        name: Skill name from list_skills (e.g. "train-forecast-model").
        section: Optional heading to return instead of the whole skill,
            matched case-insensitively as a substring of a heading
            (e.g. "Section 3" or "Tool cheat sheet"). Use list_skills'
            ``sections`` to see what is available.
        file: Optional supporting file from the skill's ``supporting_files``
            list (e.g. "references/ecm-catalog.md"). Mutually exclusive
            with ``section``.

    Returns:
        Dict with ``name``, ``description``, ``sections``, ``supporting_files``
        and ``content`` (markdown). With ``file``, ``filename`` and
        ``content`` of that file.
    """
    try:
        if section and file:
            return create_error_response("Pass either `section` or `file`, not both.")
        if file:
            return create_success_response(**_get_skill_file(name, file))
        return create_success_response(**_get_skill(name, section=section))
    except SkillNotFoundError as e:
        return create_error_response(str(e))
    except Exception as e:
        logger.exception("get_skill failed")
        return create_error_response(f"Failed to load skill: {e}")
