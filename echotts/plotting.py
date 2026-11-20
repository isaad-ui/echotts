"""Matplotlib figures used by the overfit script and TensorBoard logging."""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (works on servers)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def plot_alignment(alignment: np.ndarray, title: str = "Attention alignment") -> Figure:
    """alignment: (T_mel, T_text). A clean diagonal = the model reads left to right."""
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.T, origin="lower", aspect="auto", interpolation="none")
    ax.set_xlabel("decoder step (mel frame)")
    ax.set_ylabel("encoder step (character)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_mels(predicted: np.ndarray, real: np.ndarray | None = None) -> Figure:
    """Predicted mel (n_mels, T) above the real one, for eyeballing."""
    rows = 1 if real is None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), squeeze=False)
    panels = [("predicted", predicted)] + ([] if real is None else [("real", real)])
    for ax, (name, mel) in zip(axes[:, 0], panels):
        im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="none")
        ax.set_title(f"{name} mel")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_loss(losses: list[float]) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    fig.tight_layout()
    return fig


def save(fig: Figure, path) -> None:
    fig.savefig(path, dpi=110)
    plt.close(fig)
