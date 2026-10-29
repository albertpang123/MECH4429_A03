"""Stand-in for Joe's fault-injection module (``src/faults/``).

The injector changes the *applied* command, not the thrust: the failed rotor
keeps receiving whatever the policy commands, but its applied effectiveness is
zero from the failure step on.  This is the interface fixed in
``docs/INTERFACE.md`` section 2.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from envs.interface import ACTION_DIM, CONTROL_HZ, Scenario


class StubFaultInjector:
    def __init__(self, scenario: Scenario) -> None:
        self.reset(scenario)

    def reset(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self._failure_step = scenario.failure_step

    def is_active(self, t_s: float) -> bool:
        if self._failure_step is None:
            return False
        return self._step_from_time(t_s) >= self._failure_step

    def apply(self, u_cmd: np.ndarray, t_s: float) -> np.ndarray:
        u = np.asarray(u_cmd, dtype=np.float64).reshape(-1)
        if u.shape != (ACTION_DIM,):
            raise ValueError(f"u_cmd must have shape ({ACTION_DIM},), got {u.shape}")
        applied = u.copy()
        if self.is_active(t_s):
            applied[self.scenario.rotor_id] *= self.scenario.effectiveness
        return applied

    def describe(self) -> dict:
        """Logging/evaluation only.  Never feed this into the observation."""
        return asdict(self.scenario)

    @staticmethod
    def _step_from_time(t_s: float) -> int:
        return int(np.floor(float(t_s) * CONTROL_HZ + 0.5))
