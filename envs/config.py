"""Config loading with inheritance.

A run config may extend one or more parent files::

    # configs/g3_hover.yaml
    extends: base.yaml
    dynamics:
      mass_kg: 0.5

The child carries only what differs from the parent, so changing a shared value
(arm geometry, reward weights, observation scales) is a one-line edit in
``configs/base.yaml`` rather than a search across every run file.

Merging is recursive for mappings.  Sequences and scalars replace the parent
value wholesale — a child list is never merged element-wise, because that would
make it impossible to shorten a list.

Always load configs with :func:`load_config`.  ``yaml.safe_load`` returns the
child file alone and silently drops every inherited value.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml

REQUIRED_SECTIONS: tuple[str, ...] = (
    "run",
    "env",
    "observation",
    "reward",
    "termination",
    "dynamics",
    "faults",
    "seeds",
)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``override`` applied recursively."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, _chain: Sequence[Path] = ()) -> Dict[str, Any]:
    """Load ``path``, resolving any ``extends`` parents first."""
    path = Path(path).resolve()
    if path in _chain:
        chain = " -> ".join(str(p) for p in (*_chain, path))
        raise ValueError(f"config inheritance cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")

    parents = raw.pop("extends", [])
    if isinstance(parents, (str, Path)):
        parents = [parents]

    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / str(parent)).resolve()
        merged = deep_merge(merged, load_config(parent_path, (*_chain, path)))

    merged = deep_merge(merged, raw)
    merged["_provenance"] = {
        "source": str(path),
        "extends": [str((path.parent / str(p)).resolve()) for p in parents],
    }
    return merged


def config_hash(config: Mapping[str, Any]) -> str:
    """Short stable digest of the resolved config, ignoring provenance.

    Two runs with the same hash used the same parameters.  This is the cheap way
    to tell, months later, whether a result belongs to the current config.
    """
    payload = {key: value for key, value in config.items() if key != "_provenance"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_override(config: Dict[str, Any], assignment: str) -> None:
    """Set one dotted path from a ``path=value`` string, in place.

    This is what lets a comparison or an ablation be one command instead of one
    YAML file per combination (``train/train_ppo.py --set``, ``train/sweep.py``).
    The value is parsed by ``yaml.safe_load``, so ``2.0`` is a float, ``true`` a
    bool, ``[64, 128]`` a list and ``null`` a null.

    A path that does not already exist is rejected.  Merging a misspelt key
    would otherwise look successful while the run quietly used the old value --
    exactly the kind of silent no-op this project keeps getting bitten by.
    """
    path, separator, raw = assignment.partition("=")
    path = path.strip()
    if not separator or not path:
        raise ValueError(f"expected path=value, got {assignment!r}")

    keys = path.split(".")
    if any(key.startswith("_") for key in keys):
        raise ValueError(f"{path}: keys starting with '_' are internal")
    node: Any = config
    walked: List[str] = []
    for key in keys[:-1]:
        if not isinstance(node, Mapping) or key not in node:
            raise ValueError(_unknown_key(path, walked, node))
        walked.append(key)
        node = node[key]
    leaf = keys[-1]
    if not isinstance(node, MutableMapping) or leaf not in node:
        raise ValueError(_unknown_key(path, keys[:-1], node))
    node[leaf] = yaml.safe_load(raw)


def _unknown_key(path: str, container: Sequence[str], node: Any) -> str:
    """Name the mapping that was searched, so the hint's keys are the right ones."""
    where = ".".join(container) or "<root>"
    known = ", ".join(sorted(str(key) for key in node)) if isinstance(node, Mapping) else None
    hint = f" (known keys: {known})" if known else f" ({type(node).__name__})"
    return f"{path}: no such key at {where}{hint}"


def unresolved_paths(config: Mapping[str, Any]) -> List[str]:
    """Dotted paths whose value is still ``null`` (a TBD owned by someone)."""
    found: List[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key).startswith("_"):
                    continue
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif node is None:
            found.append(prefix)

    walk(config, "")
    return found


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"config section '{key}' must be a mapping, got {type(value).__name__}")
    return value


def _impl(config: Mapping[str, Any]) -> str:
    """``env.impl`` with its default applied, from the one place that owns it.

    Imported inside the function so that ``envs.config`` stays importable
    without pulling in ``envs.env_factory``'s lazily-loaded branch, and so the
    default is not written down twice.
    """
    from envs.env_factory import impl_name

    return impl_name(config)


def validate_structure(config: Mapping[str, Any], *, expect_resolved: bool = False) -> None:
    """Fail fast on a malformed or incomplete config.

    ``expect_resolved=True`` is for anything that will actually run: it rejects
    leftover ``null`` vehicle parameters so a placeholder config cannot start a
    long training run.  Only ``dynamics`` is checked — a ``null`` elsewhere (for
    example ``termination.tilt_limit_deg``) can mean "deliberately unused", and
    that is a decision the code should not second-guess.

    ``env.impl: gpd`` is exempt from that check: the vehicle comes from gpd's
    ``cf2x.urdf``, so ``dynamics.*`` is not read at all and a gpd config is
    allowed to leave it null.  For ``legacy`` nothing changes.
    """
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise KeyError(f"config is missing section(s): {', '.join(missing)}")
    for section in REQUIRED_SECTIONS:
        _require_mapping(config, section)

    env = config["env"]
    for key in ("physics_hz", "control_hz", "episode_length_s"):
        if env.get(key) is None:
            raise KeyError(f"env.{key} must be set")
    physics_hz = float(env["physics_hz"])
    control_hz = float(env["control_hz"])
    if physics_hz <= 0.0 or control_hz <= 0.0:
        raise ValueError("env.physics_hz and env.control_hz must be positive")
    ratio = physics_hz / control_hz
    if not float(ratio).is_integer():
        raise ValueError(f"env.physics_hz / env.control_hz must be an integer, got {ratio}")

    observation = config["observation"]
    if int(observation.get("history_len", 0)) < 1:
        raise ValueError("observation.history_len must be >= 1")
    if not isinstance(observation.get("scales"), Mapping):
        raise TypeError("observation.scales must be a mapping of field name -> scale")

    if expect_resolved and _impl(config) != "gpd":
        unresolved = [
            path for path in unresolved_paths(config) if path.startswith("dynamics.")
        ]
        if unresolved:
            raise ValueError(
                "config still has unresolved vehicle parameters: "
                + ", ".join(unresolved)
                + " — fill these in from the vehicle data before running"
            )
