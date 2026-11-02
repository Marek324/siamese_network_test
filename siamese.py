# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "torch",
#     "torchvision",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import datasets

    return F, datasets, mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Siamese network on MNIST

    A siamese network does not predict a class. It learns an **embedding**: a function
    $f(x)$ that maps an image to a point in space, so that

    - two images of the **same** digit land **close** together,
    - two images of **different** digits land **at least `margin` apart**.

    It is trained on **pairs**. Both images of a pair go through the *same* encoder (shared
    weights, hence "siamese"), and the loss looks only at the distance between the outputs:

    $$
    \mathcal{L} = y \cdot d^2 + (1 - y) \cdot \max(0,\ m - d)^2,
    \qquad d = \lVert f(x_1) - f(x_2) \rVert
    $$

    where $y = 1$ for a matching pair and $m$ is the margin.

    The embedding here is **2-D**, so we can plot it directly. Some digits are **hidden during
    training**, to check whether the network can still separate classes it has never seen.
    """)
    return


@app.cell
def _(torch):
    def pick_device():
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
                return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = pick_device()
    device
    return (device,)


@app.cell
def _(datasets, device, torch):
    MEAN, STD = 0.1307, 0.3081

    def load(train):
        ds = datasets.MNIST("data", train=train, download=True)
        x = ds.data.float().div(255).sub(MEAN).div(STD).unsqueeze(1)  # (N, 1, 28, 28)
        return x.to(device), ds.targets.to(device)

    X_train, y_train = load(train=True)
    X_test, y_test = load(train=False)
    return MEAN, STD, X_test, X_train, y_test, y_train


@app.cell
def _(torch):
    def make_pair_sampler(y, classes):
        """Return sample(n) -> (idx_a, idx_b, same), drawing 50% matching pairs.

        Only images whose label is in `classes` are used. Fully vectorized, so it
        runs on the GPU without a DataLoader.
        """
        classes = torch.as_tensor(classes, device=y.device)
        keep = torch.isin(y, classes).nonzero().squeeze(1)
        y_keep = y[keep]
        # Group the kept indices by class: order[offsets[c] : offsets[c] + counts[c]]
        order = keep[torch.argsort(y_keep)]
        counts = torch.stack([(y_keep == c).sum() for c in classes])
        offsets = torch.cumsum(counts, 0) - counts
        lut = torch.full((10,), -1, device=y.device, dtype=torch.long)
        lut[classes] = torch.arange(len(classes), device=y.device)
        k = len(classes)

        def sample(n):
            a = keep[torch.randint(len(keep), (n,), device=y.device)]
            same = torch.rand(n, device=y.device) < 0.5
            pos_a = lut[y[a]]
            # Different class: shift by 1..k-1 positions so it never wraps back to itself.
            shift = torch.randint(1, max(k, 2), (n,), device=y.device)
            pos_b = torch.where(same, pos_a, (pos_a + shift) % k)
            r = (torch.rand(n, device=y.device) * counts[pos_b]).long()
            b = order[offsets[pos_b] + r]
            return a, b, same.float()

        return sample

    return (make_pair_sampler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Training pairs

    A few random pairs. Green = same digit (target distance 0), red = different
    (target distance ≥ margin).
    """)
    return


@app.cell
def _(MEAN, STD, X_train, make_pair_sampler, plt, torch, y_train):
    def _show():
        torch.manual_seed(1)
        a, b, same = make_pair_sampler(y_train, list(range(10)))(8)
        fig, axes = plt.subplots(2, 8, figsize=(10, 3))
        for j in range(8):
            for row, idx in enumerate((a[j], b[j])):
                ax = axes[row, j]
                ax.imshow((X_train[idx, 0] * STD + MEAN).cpu(), cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                color = "tab:green" if same[j] else "tab:red"
                for s in ax.spines.values():
                    s.set_color(color)
                    s.set_linewidth(3)
            axes[0, j].set_title("same" if same[j] else "different", fontsize=9)
        fig.tight_layout()
        return fig

    _show()
    return


@app.cell
def _(nn):
    class Encoder(nn.Module):
        """Small CNN: 28x28 image -> `dim`-dimensional embedding."""

        def __init__(self, dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 14x14
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 7x7
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, dim),
            )

        def forward(self, x):
            return self.net(x)

    class Siamese(nn.Module):
        """Both inputs go through the *same* encoder: that's the whole trick."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x1, x2):
            return self.encoder(x1), self.encoder(x2)

    return Encoder, Siamese


@app.cell
def _(F, torch):
    def contrastive_loss(z1, z2, same, margin):
        d = F.pairwise_distance(z1, z2)
        pull = same * d.pow(2)  # same digit: pull together
        push = (1 - same) * torch.clamp(margin - d, min=0).pow(
            2
        )  # different: push apart
        return (pull + push).mean()

    return (contrastive_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train

    Pick which digits to hide from training, then press **Train**. Changing a setting
    afterwards resets things until you train again.
    """)
    return


@app.cell
def _(mo):
    holdout = mo.ui.multiselect(
        options=[str(i) for i in range(10)],
        value=["8", "9"],
        label="Digits hidden during training",
    )
    epochs = mo.ui.slider(1, 20, value=5, label="Epochs", show_value=True)
    margin = mo.ui.slider(
        0.5, 5.0, step=0.5, value=2.0, label="Margin", show_value=True
    )
    train_button = mo.ui.run_button(label="Train")
    mo.vstack([holdout, epochs, margin, train_button])
    return epochs, holdout, margin, train_button


@app.cell
def _(
    Encoder,
    Siamese,
    X_train,
    contrastive_loss,
    device,
    epochs,
    holdout,
    make_pair_sampler,
    margin,
    mo,
    torch,
    train_button,
    y_train,
):
    mo.stop(not train_button.value, mo.md("*Press **Train** to start.*"))

    seen_digits = [d for d in range(10) if str(d) not in holdout.value]
    unseen_digits = [d for d in range(10) if str(d) in holdout.value]

    torch.manual_seed(0)
    model = Siamese(Encoder(dim=2)).to(device)
    _opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _sample = make_pair_sampler(y_train, seen_digits)
    _batch = 256
    _steps = (
        int(torch.isin(y_train, torch.tensor(seen_digits, device=device)).sum())
        // _batch
    )

    losses = []
    model.train()
    for _ in mo.status.progress_bar(
        range(epochs.value), title="Training", show_eta=True
    ):
        for _ in range(_steps):
            _a, _b, _same = _sample(_batch)
            _z1, _z2 = model(X_train[_a], X_train[_b])
            _loss = contrastive_loss(_z1, _z2, _same, margin.value)
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            losses.append(_loss.item())
    model.eval()
    return losses, model, seen_digits, unseen_digits


@app.cell
def _(losses, np, plt):
    def _plot():
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(losses, alpha=0.3, label="per batch")
        k = 50
        if len(losses) > k:
            smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
            ax.plot(range(k - 1, len(losses)), smooth, label=f"moving avg ({k})")
        ax.set_xlabel("step")
        ax.set_ylabel("contrastive loss")
        ax.legend()
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The embedding space

    Every test image, encoded to 2-D. The network was never told what the digits are,
    only "same" or "different", yet the classes form clusters. Hidden digits are drawn
    as **×**. If they still form their own cluster, the network learned a general notion
    of "looks alike", not just the training classes.
    """)
    return


@app.cell
def _(X_test, model, torch):
    with torch.no_grad():
        Z_test = (
            torch.cat([model.encoder(_x) for _x in X_test.split(2000)]).cpu().numpy()
        )
    return (Z_test,)


@app.cell
def _(Z_test, plt, unseen_digits, y_test):
    def _plot():
        y = y_test.cpu().numpy()
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(8, 7))
        for d in range(10):
            pts = Z_test[y == d]
            hidden = d in unseen_digits
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=6 if not hidden else 10,
                marker="x" if hidden else "o",
                alpha=0.5,
                color=cmap(d),
                label=f"{d}{' (hidden)' if hidden else ''}",
            )
        ax.legend(markerscale=3, loc="best", fontsize=9)
        ax.set_title("Test set embeddings")
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Distances as a "same / different" classifier

    Sample random test pairs and look at the distance between their embeddings. A good
    model separates the two histograms, and a single distance threshold then decides
    "same digit?". This is split into pairs of **seen** digits and pairs of **hidden** ones.
    """)
    return


@app.cell
def _(Z_test, make_pair_sampler, np, plt, seen_digits, torch, unseen_digits, y_test):
    def _pair_distances(classes, n=5000):
        torch.manual_seed(2)
        a, b, same = make_pair_sampler(y_test, classes)(n)
        a, b, same = a.cpu().numpy(), b.cpu().numpy(), same.cpu().numpy().astype(bool)
        return np.linalg.norm(Z_test[a] - Z_test[b], axis=1), same

    def _best_threshold(d, same):
        ts = np.quantile(d, np.linspace(0, 1, 200))
        accs = [((d < t) == same).mean() for t in ts]
        i = int(np.argmax(accs))
        return ts[i], accs[i]

    groups = [("seen digits", seen_digits)]
    if len(unseen_digits) >= 2:
        groups.append(("hidden digits", unseen_digits))

    _fig, _axes = plt.subplots(
        1, len(groups), figsize=(6 * len(groups), 3.5), squeeze=False
    )
    _d_seen, _same_seen = _pair_distances(seen_digits)
    threshold, _ = _best_threshold(_d_seen, _same_seen)
    for _ax, (_name, _classes) in zip(_axes[0], groups):
        _d, _same = _pair_distances(_classes)
        _acc = ((_d < threshold) == _same).mean()
        _bins = np.linspace(0, np.quantile(_d, 0.99), 60)
        _ax.hist(
            _d[_same], bins=_bins, alpha=0.6, label="same digit", color="tab:green"
        )
        _ax.hist(_d[~_same], bins=_bins, alpha=0.6, label="different", color="tab:red")
        _ax.axvline(threshold, color="k", ls="--", label=f"threshold {threshold:.2f}")
        _ax.set_title(f"{_name}: accuracy {_acc:.1%}")
        _ax.set_xlabel("embedding distance")
        _ax.legend()
    _fig.tight_layout()
    _fig
    return (threshold,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Try it yourself

    Pick any two test images. The model encodes both and compares the distance to the
    threshold chosen above (on seen digits).
    """)
    return


@app.cell
def _(mo):
    idx_a = mo.ui.number(0, 9999, value=0, label="Image A (test index)")
    idx_b = mo.ui.number(0, 9999, value=1, label="Image B (test index)")
    mo.hstack([idx_a, idx_b], justify="start")
    return idx_a, idx_b


@app.cell
def _(MEAN, STD, X_test, Z_test, idx_a, idx_b, np, plt, threshold, y_test):
    def _compare(i, j):
        d = float(np.linalg.norm(Z_test[i] - Z_test[j]))
        verdict = "same" if d < threshold else "different"
        truth = "same" if y_test[i] == y_test[j] else "different"
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.8))
        for ax, k in zip(axes, (i, j)):
            ax.imshow((X_test[k, 0] * STD + MEAN).cpu(), cmap="gray")
            ax.set_title(f"#{k} (label {int(y_test[k])})")
            ax.axis("off")
        ok = "✓" if verdict == truth else "✗"
        fig.suptitle(f"distance {d:.2f} → model says {verdict} {ok}")
        fig.tight_layout()
        return fig

    _compare(idx_a.value, idx_b.value)
    return


if __name__ == "__main__":
    app.run()
