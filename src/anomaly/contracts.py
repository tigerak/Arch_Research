from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol

AnomalyRunResult = Dict[str, Any]


class AnomalyMethodRunner(Protocol):
    def __call__(self, cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> AnomalyRunResult:
        ...