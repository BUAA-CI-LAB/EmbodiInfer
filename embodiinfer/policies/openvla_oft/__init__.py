"""OpenVLA-OFT adapter (RLinf discrete-token variant) -> embodiinfer ``VLAPolicy``.

The first non-flow first-class policy: a single causal forward (parallel decoding)
producing a chunk of discrete action tokens (256-bin categorical), aligned bit-exact
against RLinf's ``openvla_oft`` rollout generator. See ``docs/proposals/0003``.

The concrete policy lives in ``modeling_openvla_oft`` and registers itself via
``@register_policy("openvla_oft")``; it is imported by ``embodiinfer.policies`` (the registry
trigger). ``head`` and ``processor_openvla_oft`` are importable standalone for testing.
"""
