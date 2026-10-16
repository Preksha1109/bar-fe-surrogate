"""Step 3: train the MLP surrogate on the FE dataset.

Split: 70 % train / 15 % validation / 15 % test (test indices saved for evaluation).
The model with the lowest validation loss is kept.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barfe.surrogate import MLP, Surrogate, encode_inputs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 3000
BATCH = 256
LR = 2e-3
SEED = 0


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    d = np.load(ROOT / "data" / "dataset.npz")
    params, u_fe = d["params"], d["u_fe"]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(params))
    n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    y_mm = u_fe[:, 1:] * 1e3                 # drop fixed node, metres -> mm
    u_mean, u_std = y_mm[tr].mean(0), y_mm[tr].std(0)
    X = torch.from_numpy(encode_inputs(params))
    Y = torch.from_numpy(((y_mm - u_mean) / u_std).astype(np.float32))

    model = MLP(n_out=Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    best, best_state, history = np.inf, None, []
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
        sched.step()

        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tr_loss = loss_fn(model(Xtr), Ytr).item()
                va_loss = loss_fn(model(Xva), Yva).item()
            history.append({"epoch": epoch, "train": tr_loss, "val": va_loss})
            if va_loss < best:
                best = va_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if epoch % 500 == 0:
                print(f"epoch {epoch:5d}  train {tr_loss:.3e}  val {va_loss:.3e}")
    print(f"Trained in {time.perf_counter() - t0:.1f} s, best val loss {best:.3e}")

    model.load_state_dict(best_state)
    (ROOT / "models").mkdir(exist_ok=True)
    Surrogate(model, u_mean, u_std).save(ROOT / "models" / "surrogate.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "split.npz", train=tr, val=va, test=te)
    (ROOT / "results" / "training_history.json").write_text(json.dumps(history, indent=1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved models/surrogate.pt ({n_params} parameters)")


if __name__ == "__main__":
    main()
