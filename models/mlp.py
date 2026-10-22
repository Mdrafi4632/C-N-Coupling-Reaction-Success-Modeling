import numpy as np

MODEL_NAME = "MLP"

_DEFAULTS = {
    "hidden_sizes": [1024, 512, 256], "dropout": 0.3,
    "lr": 1e-3, "epochs": 100, "batch_size": 256, "patience": 20,
}

def train(X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    p      = {**_DEFAULTS, **cfg.get("mlp_params", {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build network
    layers, in_dim = [], X_tr.shape[1]
    for h in p["hidden_sizes"]:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(p["dropout"])]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers).to(device)

    opt       = optim.Adam(model.parameters(), lr=p["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn   = nn.MSELoss()

    def to_t(arr): return torch.tensor(arr, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(to_t(X_tr), to_t(y_tr)),
                        batch_size=p["batch_size"], shuffle=True)
    X_val_t, y_val_t, X_te_t = to_t(X_val), to_t(y_val), to_t(X_te)

    best_loss, best_state, no_improve = float("inf"), None, 0
    for epoch in range(p["epochs"]):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t).squeeze(-1), y_val_t).item()
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= p["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = model(X_val_t).squeeze(-1).cpu().numpy().astype(np.float32)
        tes_p = model(X_te_t).squeeze(-1).cpu().numpy().astype(np.float32)
    return val_p, tes_p
