"""Automated CI smoke tests for DreamerV3 world model, replay, and drivers."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import warnings

import elements
import embodied
from embodied.core import mjx_driver, selectors, streams
from embodied.envs import from_gym
from embodied.jax import heads, outs
import gymnasium as gym
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import ruamel.yaml as yaml

from dreamerv3.agent import Agent


def test_agent_train_report_and_zero_deprecations(tmp_path: Path) -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "dreamerv3" / "configs.yaml"
    cfgs = yaml.YAML(typ="safe").load(cfg_path.read_text(encoding="utf-8"))
    cfg = (
        elements.Config(cfgs["defaults"])
        .update(cfgs["debug"])
        .update({"logdir": str(tmp_path / "run"), "jax.platform": "cpu"})
    )
    obs_space = {
        "proprio": elements.Space(np.float32, (6,)),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act_space = {"action": elements.Space(np.float32, (3,), -1.0, 1.0)}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        agent = Agent(
            obs_space,
            act_space,
            elements.Config(
                **cfg.agent,
                logdir=cfg.logdir,
                seed=0,
                jax=cfg.jax,
                batch_size=cfg.batch_size,
                batch_length=cfg.batch_length,
                replay_context=cfg.replay_context,
                report_length=cfg.report_length,
                replica=0,
                replicas=1,
            ),
        )
        b, t = cfg.batch_size, cfg.batch_length + cfg.replay_context
        rng = np.random.default_rng(0)
        raw_batch = {
            "proprio": rng.standard_normal((b, t, 6)).astype(np.float32),
            "action": rng.uniform(-1, 1, (b, t, 3)).astype(np.float32),
            "reward": rng.standard_normal((b, t)).astype(np.float32),
            "is_first": np.zeros((b, t), dtype=bool),
            "is_last": np.zeros((b, t), dtype=bool),
            "is_terminal": np.zeros((b, t), dtype=bool),
            "consec": np.zeros((b, t), dtype=np.int32),
            "stepid": rng.integers(0, 255, (b, t, 20), dtype=np.uint8),
            "dyn/deter": np.zeros((b, t, 8), dtype=np.float32),
            "dyn/stoch": np.zeros((b, t, 2, 4), dtype=np.float32),
        }
        raw_batch["is_first"][:, 0] = True
        tr_stream = iter(agent.stream(streams.Stateless(lambda: dict(raw_batch))))
        carry_tr = agent.init_train(b)
        carry_tr, _, _ = agent.train(carry_tr, next(tr_stream))
        carry_tr, _, train_mets = agent.train(carry_tr, next(tr_stream))
        assert "opt/loss" in train_mets

        rep_t = cfg.report_length + cfg.replay_context
        rep_raw = {k: v[:, :rep_t] for k, v in raw_batch.items()}
        rep_stream = iter(agent.stream(streams.Stateless(lambda: dict(rep_raw))))
        carry_rep = agent.init_report(b)
        _, rep_mets = agent.report(carry_rep, next(rep_stream))
        assert len(rep_mets) > 0

    bad = [
        str(w.message)
        for w in caught
        if "set_mesh" in str(w.message) or "global_norm" in str(w.message)
    ]
    assert bad == []


def test_distributions_and_heads() -> None:
    with jax.transfer_guard("allow"):
        bin_dist = outs.Binary(jnp.zeros((2, 3), dtype=jnp.float32))
        sample = bin_dist.sample(jax.random.PRNGKey(0))
        assert sample.shape == (2, 3)

        agg_bin = outs.Agg(
            outs.Binary(jnp.zeros((2, 4), dtype=jnp.float32)), 1, jnp.sum
        )
        prob = agg_bin.prob(jnp.ones((2, 4), dtype=jnp.float32))
        np.testing.assert_allclose(np.asarray(prob), np.full((2,), 0.5**4), atol=1e-6)

        def build_twohot(x):
            h = heads.Head(
                elements.Space(np.float32, ()), "symexp_twohot", bins=8, name="h"
            )
            return h(x).bins

        pure = nj.pure(build_twohot)
        params, _ = pure(
            {},
            jnp.zeros((2, 4), dtype=jnp.bfloat16),
            seed=jax.random.PRNGKey(0),
            create=True,
        )
        _, bins = pure(
            params, jnp.zeros((2, 4), dtype=jnp.bfloat16), seed=jax.random.PRNGKey(0)
        )
        assert len(np.unique(np.asarray(bins))) == 8


def test_replay_selectors_and_streams(tmp_path: Path) -> None:
    sel = selectors.Mixture(
        {
            "u": selectors.Uniform(seed=0),
            "r": selectors.Recency(0.99 ** np.arange(32), seed=0),
        },
        {"u": 0.5, "r": 0.5},
    )
    replay = embodied.replay.Replay(
        length=4, capacity=32, directory=tmp_path / "replay", chunksize=8, selector=sel
    )
    for step in range(20):
        replay.add({"obs": np.float32(step), "is_first": step == 0}, worker=0)
    assert len(sel) > 0
    seq = replay.sample(2, "train")
    assert seq["obs"].shape == (2, 4)

    s1 = streams.Stateless(lambda: {"v": np.ones((2, 4), np.float32)})
    mixed = iter(streams.Mixer({"a": s1, "b": s1}, {"a": 0.5, "b": 0.5}))
    assert next(mixed)["v"].shape == (2, 4)


def test_from_gym_and_mjx_driver() -> None:
    env = from_gym.FromGym(gym.make("CartPole-v1"))
    try:
        obs0 = env.step({"action": np.int32(0), "reset": np.bool_(True)})
        assert bool(obs0["is_first"]) is True
        obs1 = env.step({"action": np.int32(1), "reset": np.bool_(False)})
        assert len(obs1) >= 4
    finally:
        env.close()

    class _ToyState(NamedTuple):
        obs: dict
        reward: jax.Array
        done: jax.Array
        metrics: dict

    class _ToyEnv:
        action_size = 2

        class _Cfg:
            max_episode_steps = 2

        _config = _Cfg()

        def reset(self, rng):
            return _ToyState(
                {"state": jnp.zeros((3,))}, jnp.float32(0.0), jnp.float32(0.0), {}
            )

        def step(self, state, action):
            nxt = state.obs["state"] + 1.0
            done = jnp.where(nxt[0] >= 2.0, jnp.float32(1.0), jnp.float32(0.0))
            return _ToyState({"state": nxt}, jnp.float32(1.0), done, {})

    drv = mjx_driver.MJXDriver(_ToyEnv(), num_envs=2, max_episode_steps=2, seed=0)
    seen = []
    drv.on_step(lambda trn, wid: seen.append((wid, trn)))
    drv.reset()
    drv(lambda c, o: (c, {"action": np.zeros((2, 2), np.float32)}, {}), steps=8)
    w0 = [t for wid, t in seen if wid == 0]
    assert [bool(t["is_last"]) for t in w0[:4]] == [False, False, True, False]
    assert float(w0[2]["state"][0]) == 2.0
