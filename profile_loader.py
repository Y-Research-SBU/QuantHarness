"""
ProfileLoader — discover and load tournament profiles.

Reads each ``instances/<name>/profile.py`` for its exported ``PROFILE``,
plus the ``instances/<name>/state.json`` heartbeat (when present), and
exposes helpers used by the dashboard's ``/api/instances`` endpoint.

Optional per-instance overrides may live in ``instances/<name>/overrides.json``;
they shallow-merge over the profile's dataclass fields.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from instances.profile import Profile

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = REPO_ROOT / "instances"


class ProfileLoader:
    """Discover instance directories and load their effective configs."""

    def __init__(self, instances_dir: Optional[Path] = None):
        self.instances_dir = Path(instances_dir) if instances_dir else INSTANCES_DIR

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[str]:
        """Return the names of every instance directory containing profile.py."""
        if not self.instances_dir.exists():
            return []
        out: List[str] = []
        for child in sorted(self.instances_dir.iterdir()):
            if child.is_dir() and (child / "profile.py").exists():
                out.append(child.name)
        return out

    # ------------------------------------------------------------------
    # Loading + merging
    # ------------------------------------------------------------------

    def load(self, name: str) -> Profile:
        """Load and merge the effective profile for ``name``.

        Raises :class:`FileNotFoundError` if the directory is missing.
        """
        profile_dir = self.instances_dir / name
        profile_path = profile_dir / "profile.py"
        if not profile_path.exists():
            raise FileNotFoundError(f"profile not found: {profile_path}")

        base = self._import_profile(name, profile_path)
        overrides = self._read_overrides(profile_dir)
        if overrides:
            return self._merge(base, overrides)
        return base

    def _import_profile(self, name: str, profile_path: Path) -> Profile:
        """Load PROFILE from ``profile_path`` by absolute file path.

        We intentionally do NOT use ``importlib.import_module('instances.X.profile')``
        because that resolves against the *repo* package, not whichever
        directory this loader was pointed at. Loading by file spec keeps
        tests (which use a tmp ``instances/`` tree) honest.
        """
        import importlib.util
        spec_name = f"_pl_{name}_{abs(hash(str(profile_path)))}"
        spec = importlib.util.spec_from_file_location(spec_name, profile_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load spec for {profile_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "PROFILE"):
            raise AttributeError(f"{profile_path} does not define PROFILE")
        return module.PROFILE

    def _read_overrides(self, profile_dir: Path) -> Dict[str, Any]:
        path = profile_dir / "overrides.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("invalid overrides at %s: %s", path, exc)
            return {}

    def _merge(self, base: Profile, overrides: Dict[str, Any]) -> Profile:
        """Shallow-merge ``overrides`` over a Profile, ignoring unknown keys."""
        valid = {f.name for f in fields(Profile)}
        kwargs = {k: v for k, v in overrides.items() if k in valid}
        # Coerce cell_overrides JSON arrays back to tuples if present.
        if "cell_overrides" in kwargs and kwargs["cell_overrides"] is not None:
            kwargs["cell_overrides"] = [tuple(c) for c in kwargs["cell_overrides"]]
        try:
            return replace(base, **kwargs)
        except TypeError as exc:
            logger.warning("could not merge overrides: %s", exc)
            return base

    # ------------------------------------------------------------------
    # State / heartbeat
    # ------------------------------------------------------------------

    def state_file(self, name: str) -> Path:
        return self.instances_dir / name / "state.json"

    def read_state(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.state_file(name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as exc:
            logger.warning("invalid state.json at %s: %s", path, exc)
            return None

    def is_alive(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            import os
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # API helper
    # ------------------------------------------------------------------

    def describe(self, name: str) -> Dict[str, Any]:
        """Return a JSON-able description of one instance for /api/instances."""
        profile = self.load(name)
        state = self.read_state(name) or {}
        pid = state.get("pid")
        running = self.is_alive(pid)
        return {
            "name": profile.name,
            "status": "running" if running else "stopped",
            "pid": pid,
            "started_at": state.get("started_at"),
            "last_heartbeat": state.get("last_heartbeat"),
            "db_path": state.get("db_path") or profile.db_path,
            "equity": state.get("equity"),
            "open_positions": state.get("open_positions"),
            "realized_pnl": state.get("realized_pnl"),
            "total_trades": state.get("total_trades"),
            "last_scan": state.get("last_scan"),
            "profile": {
                "universe": profile.universe,
                "asset_categories": profile.asset_categories,
                "strategies": profile.strategies,
                "cell_overrides": [list(c) for c in profile.cell_overrides],
                "position_size_multiplier": profile.position_size_multiplier,
                "entry_threshold_multiplier": profile.entry_threshold_multiplier,
                "regime_filter_enabled": profile.regime_filter_enabled,
                "use_l0_whitelist": profile.use_l0_whitelist,
                "use_l1_evolution": profile.use_l1_evolution,
            },
        }

    def describe_all(self) -> List[Dict[str, Any]]:
        out = []
        for name in self.list_profiles():
            try:
                out.append(self.describe(name))
            except Exception as exc:
                logger.warning("describe(%s) failed: %s", name, exc)
                out.append({"name": name, "status": "error", "error": str(exc)})
        return out


__all__ = ["ProfileLoader"]
