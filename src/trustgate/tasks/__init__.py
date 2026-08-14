"""Task-type schema validation.

The :class:`~trustgate.models.Item` model guarantees the *envelope* (id, task_type, ...).
These validators enforce the *payload* shape each task type requires, so a malformed seed
case fails loudly at load time rather than deep inside an evaluator.
"""

from __future__ import annotations

from trustgate.models import Item, TaskType


class SchemaError(ValueError):
    """Raised when an item's input/references payload violates its task-type schema."""


def _require(condition: bool, item_id: str, message: str) -> None:
    if not condition:
        raise SchemaError(f"[{item_id}] {message}")


def _validate_generation(item: Item) -> None:
    _require(isinstance(item.input, str) and item.input.strip() != "", item.id,
             "generation.input must be a non-empty string")
    expected = item.metadata.get("expected", "answer")
    if expected == "answer":
        _require(item.references is not None, item.id,
                 "generation.references required unless metadata.expected == 'abstain'")


def _validate_retrieval(item: Item) -> None:
    inp = item.input
    _require(isinstance(inp, dict), item.id, "retrieval.input must be an object")
    _require("question" in inp and isinstance(inp["question"], str), item.id,
             "retrieval.input.question (string) is required")
    _require("corpus" in inp and isinstance(inp["corpus"], list) and len(inp["corpus"]) > 0,
             item.id, "retrieval.input.corpus (non-empty list) is required")
    corpus_ids = set()
    for doc in inp["corpus"]:
        _require(isinstance(doc, dict) and "id" in doc and "text" in doc, item.id,
                 "each corpus doc needs 'id' and 'text'")
        corpus_ids.add(doc["id"])
    refs = item.references
    _require(isinstance(refs, dict), item.id, "retrieval.references must be an object")
    _require("answer" in refs and "supporting_ids" in refs, item.id,
             "retrieval.references needs 'answer' and 'supporting_ids'")
    _require(isinstance(refs["supporting_ids"], list), item.id,
             "retrieval.references.supporting_ids must be a list")
    for sid in refs["supporting_ids"]:
        _require(sid in corpus_ids, item.id,
                 f"supporting id '{sid}' not present in corpus")


def _validate_tool_trajectory(item: Item) -> None:
    inp = item.input
    _require(isinstance(inp, dict), item.id, "tool_trajectory.input must be an object")
    _require("goal" in inp and isinstance(inp["goal"], str), item.id,
             "tool_trajectory.input.goal (string) is required")
    _require("tools" in inp and isinstance(inp["tools"], list) and len(inp["tools"]) > 0,
             item.id, "tool_trajectory.input.tools (non-empty list) is required")
    tool_names = set()
    for t in inp["tools"]:
        _require(isinstance(t, dict) and "name" in t, item.id, "each tool needs a 'name'")
        tool_names.add(t["name"])
    refs = item.references
    _require(isinstance(refs, dict) and "required_calls" in refs, item.id,
             "tool_trajectory.references.required_calls is required")
    for call in refs["required_calls"]:
        _require(isinstance(call, dict) and "tool" in call, item.id,
                 "each required_call needs a 'tool'")
        _require(call["tool"] in tool_names, item.id,
                 f"required_call tool '{call['tool']}' is not in the item's tool list")


_VALIDATORS = {
    TaskType.GENERATION: _validate_generation,
    TaskType.RETRIEVAL: _validate_retrieval,
    TaskType.TOOL_TRAJECTORY: _validate_tool_trajectory,
}


def validate_item(item: Item) -> None:
    """Validate an item against its task-type schema. Raises :class:`SchemaError`.

    Task types without a dedicated validator (e.g. structured, pairwise) pass through
    until their milestone adds one.
    """
    validator = _VALIDATORS.get(item.task_type)
    if validator is not None:
        validator(item)


__all__ = ["SchemaError", "validate_item"]
