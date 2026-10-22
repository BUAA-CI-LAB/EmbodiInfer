"""What the coding agent of a tuning or registration flow may touch.

Humanize2 holds a session to anything narrower than every scope ``ALL`` with an
OS fence that needs Linux Landlock. Roles are therefore declared with every
scope, so an explicitly unfenced run can start where Landlock is unavailable;
every other run narrows the agent back to Humanize2's default fence before its
first session opens. Only modules that already require hmz import this one.
"""

from __future__ import annotations

from hmz.flows import Agent, Permission, PermissionKind

from scripts.kernel_tuning.humanize_compat import configure_humanize

#: Humanize2's default grant: write the session workdir, read the rest, stay online.
FENCED = Permission()
#: Nothing fenced. Isolation is then the operator's account or container.
UNFENCED = Permission(
    local=PermissionKind.ALL,
    user=PermissionKind.ALL,
    system=PermissionKind.ALL,
    online=PermissionKind.ALL,
)


class Coder(Agent):
    """A coding-agent role that may run unfenced; ``session_agent`` narrows it otherwise."""

    _permission = UNFENCED


def session_agent(agent: Agent, *, unfenced: bool) -> Agent:
    """Return the agent sessions must be opened with for the run's recorded fence choice.

    Fenced runs keep the exact permission the flows used before this option existed.
    """
    configure_humanize()
    return agent if unfenced else agent.derive(permission=FENCED)
