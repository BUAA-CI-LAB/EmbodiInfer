from types import SimpleNamespace

import pytest
import torch

from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy


def schedule(rho, steps):
    return Pi05Policy.flow_schedule(SimpleNamespace(flow_schedule_rho=rho), steps)


def test_default_schedule_keeps_uniform_steps():
    assert schedule(1.0, 4) == [(1.0, -0.25), (0.75, -0.25), (0.5, -0.25), (0.25, -0.25)]


@pytest.mark.parametrize(("rho", "steps"), [(2.0, 3), (3.0, 2), (0.5, 4)])
def test_power_schedule_runs_from_noise_to_data(rho, steps):
    rows = schedule(rho, steps)
    times = [time for time, _ in rows] + [rows[-1][0] + rows[-1][1]]
    assert times == pytest.approx([(1 - index / steps) ** rho for index in range(steps + 1)])
    assert times[0] == 1.0 and times[-1] == pytest.approx(0.0, abs=1e-15)
    sizes = [-dt for _, dt in rows]
    assert all(size > 0 for size in sizes)
    # rho > 1 takes its small steps at the data end (t = 0), rho < 1 at the noise end.
    assert sizes == sorted(sizes, reverse=rho > 1)


@pytest.mark.parametrize("rho", [0.0, -1.0, float("nan")])
def test_schedule_rejects_non_positive_rho(rho):
    with pytest.raises(ValueError, match="positive"):
        schedule(rho, 3)


def test_graph_variant_separates_schedules():
    prefix = SimpleNamespace(prefix_pad_masks=torch.zeros(1, 7), all_valid=True)
    native = SimpleNamespace(native_inference=True, flow_schedule_rho=3.0)
    assert Pi05Policy.cuda_graph_variant(native, prefix) == (7, True, 3.0)
    eager = SimpleNamespace(native_inference=False, flow_schedule_rho=1.0)
    assert Pi05Policy.cuda_graph_variant(eager, prefix) == (True, 1.0)
