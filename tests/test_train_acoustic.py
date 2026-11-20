"""Checkpoint helpers and the resume behaviour of the training loop (tiny model, CPU)."""
import copy

import torch

from echotts.checkpoint import latest_checkpoint, list_checkpoints, save_checkpoint
from echotts.train_acoustic import train


def test_checkpoints_keep_last_n(tmp_path) -> None:
    for step in (1000, 2000, 3000, 4000):
        save_checkpoint(tmp_path, step, {"x": torch.tensor(step)}, keep=3)
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["step_000002000.pt", "step_000003000.pt", "step_000004000.pt"]
    assert latest_checkpoint(tmp_path).name == "step_000004000.pt"


def _tiny_cfg(cfg: dict) -> dict:
    """Shrink the model so a few CPU training steps take seconds."""
    c = copy.deepcopy(cfg)
    m = c["model"]
    m["embedding_dim"] = 16
    m["encoder"].update(channels=16, lstm_hidden=8)
    m["attention"].update(dim=8, location_filters=4, location_kernel=5)
    m["prenet"]["sizes"] = [16, 16]
    m["decoder"].update(lstm_units=16, max_decoder_steps=10)
    m["postnet"]["channels"] = 16
    c["training"].update(checkpoint_every=2, log_every=1, keep_checkpoints=3,
                         split={"train": 2, "val": 1, "test": 0, "seed": 0})
    c["data"]["num_workers"] = 0
    c["audio"]["griffin_lim_iters"] = 2
    return c


def test_training_resumes_from_same_step(fake_ljspeech, cfg, tmp_path) -> None:
    c = _tiny_cfg(cfg)
    ckpt, logs = tmp_path / "ckpt", tmp_path / "logs"
    assert train(c, fake_ljspeech, ckpt, logs, max_steps=2, batch_size=2) == 2
    assert latest_checkpoint(ckpt).name == "step_000000002.pt"
    # "Kill" and restart with a higher target: it must continue from step 2, not 0.
    assert train(c, fake_ljspeech, ckpt, logs, max_steps=4, batch_size=2) == 4
    state = torch.load(latest_checkpoint(ckpt), weights_only=True)
    assert state["step"] == 4 and "optimizer" in state
