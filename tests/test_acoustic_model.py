"""Shape and behaviour tests for every acoustic-model piece on tiny fake batches (CPU)."""
import copy

import pytest
import torch

from echotts.losses import TacotronLoss, masked_mse
from echotts.models.attention import LocationSensitiveAttention
from echotts.models.decoder import Decoder
from echotts.models.encoder import Encoder
from echotts.models.layers import lengths_to_mask
from echotts.models.postnet import Postnet
from echotts.models.tacotron import Tacotron, count_parameters

B, T_TEXT, T_MEL = 3, 11, 17


def _text_batch():
    lengths = torch.tensor([11, 8, 5])
    text = torch.randint(2, 30, (B, T_TEXT))
    text[lengths.unsqueeze(1) <= torch.arange(T_TEXT)] = 0
    return text, lengths


def test_encoder_shape(cfg) -> None:
    text, lengths = _text_batch()
    out = Encoder(40, cfg["model"]).eval()(text, lengths)
    assert out.shape == (B, T_TEXT, 512)
    assert (out[2, 5:] == 0).all()                       # packed LSTM leaves padding at zero


def test_attention_sums_to_one_and_masks(cfg) -> None:
    att = LocationSensitiveAttention(1024, 512, cfg["model"]["attention"])
    memory = torch.randn(B, T_TEXT, 512)
    mask = lengths_to_mask(torch.tensor([11, 8, 5]), T_TEXT)
    ctx, w = att(torch.randn(B, 1024), memory, att.process_memory(memory),
                 torch.zeros(B, 2, T_TEXT), mask)
    assert ctx.shape == (B, 512) and w.shape == (B, T_TEXT)
    assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5)
    assert (w[~mask] == 0).all()


@pytest.mark.parametrize("r", [1, 2, 3])
def test_decoder_both_modes(cfg, r) -> None:
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["decoder"]["frames_per_step"] = r
    dec = Decoder(model_cfg, 80, 512)
    memory = torch.randn(B, T_TEXT, 512)
    mask = lengths_to_mask(torch.tensor([11, 8, 5]), T_TEXT)
    n_steps = -(-T_MEL // r)                                  # T_MEL=17 is not a multiple of 2 or 3
    mel, stop, align = dec(memory, torch.randn(B, 80, T_MEL), mask)
    assert mel.shape == (B, 80, T_MEL) and stop.shape == (B, T_MEL)
    assert align.shape == (B, n_steps, T_TEXT)
    dec.eval()
    mel, stop, align, hit_max = dec.infer(memory[:1], max_steps=7)
    assert mel.shape[:2] == (1, 80) and mel.shape[2] % r == 0 and r <= mel.shape[2] <= 7 + r - 1
    assert align.shape == (1, mel.shape[2] // r, T_TEXT)


def test_teacher_forcing_feeds_last_frame_of_each_group(cfg) -> None:
    """With r=2, step t must see real frame 2t-1 (the last of the previous pair)."""
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["decoder"]["frames_per_step"] = 2
    dec = Decoder(model_cfg, 80, 512)
    seen = []
    dec.prenet.forward = lambda x: (seen.append(x.clone()), torch.zeros(*x.shape[:-1], 256))[1]
    target = torch.arange(6.).view(1, 1, 6).expand(1, 80, 6).contiguous()   # frame i has value i
    dec(torch.randn(1, 4, 512), target, None)
    assert seen[0][0, :, 0].tolist() == [0.0, 1.0, 3.0]       # go frame (zeros), frame 1, frame 3


def test_postnet_shape(cfg) -> None:
    assert Postnet(80, cfg["model"]["postnet"])(torch.randn(B, 80, T_MEL)).shape == (B, 80, T_MEL)


def test_full_model_and_param_count(cfg) -> None:
    model = Tacotron(cfg)
    text, lengths = _text_batch()
    out = model(text, lengths, torch.randn(B, 80, T_MEL))
    assert out["mel_after"].shape == (B, 80, T_MEL)
    r = cfg["model"]["decoder"]["frames_per_step"]
    assert out["alignments"].shape == (B, -(-T_MEL // r), T_TEXT)
    n = count_parameters(model)
    assert 26e6 < n < 30e6, n                            # roughly 28 million
    model.eval()
    res = model.infer(text[:1], max_steps=5)
    assert res["mel"].shape[:2] == (1, 80)


def test_loss_ignores_padded_frames(cfg) -> None:
    target = torch.randn(B, 80, T_MEL)
    lengths = torch.tensor([17, 10, 4])
    pred = target.clone() + 0.1
    base = masked_mse(pred, target, lengths)
    garbage = pred.clone()
    garbage[1, :, 10:] = 1000.0                           # change only padded frames
    garbage[2, :, 4:] = -1000.0
    assert torch.allclose(masked_mse(garbage, target, lengths), base)

    out = {"mel_before": pred, "mel_after": pred, "stop_logits": torch.zeros(B, T_MEL)}
    batch = {"mel": target, "mel_lengths": lengths, "stop": torch.zeros(B, T_MEL)}
    out2 = copy.copy(out); out2["mel_after"] = garbage
    loss = TacotronLoss(1.0)
    assert torch.allclose(loss(out, batch)["total"], loss(out2, batch)["total"])


def test_one_backward_step_reduces_loss(cfg) -> None:
    torch.manual_seed(0)
    small = copy.deepcopy(cfg)
    small["model"]["prenet"]["dropout"] = 0.0
    for part in ("encoder", "postnet"):
        small["model"][part]["dropout"] = 0.0
    model = Tacotron(small)
    text, lengths = _text_batch()
    mel = torch.randn(B, 80, T_MEL)
    batch = {"mel": mel, "mel_lengths": torch.tensor([17, 12, 9]), "stop": torch.zeros(B, T_MEL)}
    loss_fn, opt = TacotronLoss(1.0), torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        loss = loss_fn(model(text, lengths, mel), batch)["total"]
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
