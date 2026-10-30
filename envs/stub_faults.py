"""Stand-in for Joe's fault-injection module (``src/faults/``).

The injector changes the *applied* command, not the thrust: the failed rotor
keeps receiving whatever the policy commands, but its applied effectiveness is
zero from the failure step on.  This is the interface fixed in
``docs/INTERFACE.md`` section 2.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from envs.interface import ACTION_DIM, Scenario


class StubFaultInjector:
    """Masks the failed rotor from ``Scenario.failure_time_s`` onwards.

    The comparison is done on time, not control steps, so the injector needs no
    copy of the control rate.  The environment quantises the failure time to the
    control grid when it samples the scenario, so this stays step-exact without
    a hidden ``CONTROL_HZ`` constant that could drift from the config.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.reset(scenario)

    def reset(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def is_active(self, t_s: float) -> bool:
        failure_time = self.scenario.failure_time_s
        if failure_time is None:
            return False
        return float(t_s) >= float(failure_time) - 1e-9

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
