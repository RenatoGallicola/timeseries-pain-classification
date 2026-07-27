"""Training loops and the subject-wise cross-validation that produces OOF logits.

Two details matter for correctness here:

* every window of a subject stays in the same fold (grouped CV), otherwise the
  model sees the same recording on both sides of the split and the validation
  score becomes meaningless;
* the reported metric is **weighted** F1, matching what the notebooks compute.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(ds: TensorDataset, batch_size: int, shuffle: bool,
                drop_last: bool, device: torch.device) -> DataLoader:
    """Build a DataLoader; when shuffling a labelled set, oversample rare classes.

    `high_pain` is 56 subjects out of 661, so uniform sampling leaves batches
    that contain none of it.
    """
    sampler = None
    if shuffle and len(ds.tensors) == 2:
        labels = ds.tensors[1].numpy()
        counts = np.bincount(labels)
        weights = 1.0 / (counts + 1e-9)
        sampler = WeightedRandomSampler(weights[labels], len(labels), replacement=True)
        shuffle = False

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=2, pin_memory=device.type == "cuda", sampler=sampler,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, all_preds, all_tgts = 0.0, [], []

    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_tgts.append(yb.cpu().numpy())

    preds, tgts = np.concatenate(all_preds), np.concatenate(all_tgts)
    return total_loss / len(loader.dataset), f1_score(tgts, preds, average="weighted")


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_tgts = 0.0, [], []

    for xb, yb in loader:
        xb, yb = xb.to(device).float(), yb.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_tgts.append(yb.cpu().numpy())

    preds, tgts = np.concatenate(all_preds), np.concatenate(all_tgts)
    return total_loss / len(loader.dataset), f1_score(tgts, preds, average="weighted")


def fit(model, train_loader, val_loader, *, device, class_weights=None,
        lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        max_epochs=200, patience=30, verbose_every=10):
    """Train with AdamW + ReduceLROnPlateau and early stopping on validation F1.

    Restores the best checkpoint before returning. Returns ``(model, history)``.
    """
    weight = None if class_weights is None else torch.tensor(
        class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, best_epoch, best_state, stale = -1.0, -1, None, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)

        if val_loader is None:                      # full-data refit: no validation
            history.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1})
            continue

        va_loss, va_f1 = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(va_f1)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_f1": tr_f1,
                        "val_loss": va_loss, "val_f1": va_f1})

        if verbose_every and epoch % verbose_every == 0:
            print(f"Epoch {epoch:3d}/{max_epochs} | Train: Loss={tr_loss:.4f}, "
                  f"F1={tr_f1:.4f} | Val: Loss={va_loss:.4f}, F1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1, best_epoch, stale = va_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best epoch = {best_epoch}, best val F1 = {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def oof_predict(build_model, sequences, labels, users, *, device, n_splits=5,
                batch_size=128, **fit_kwargs):
    """Grouped K-fold that returns out-of-fold window-level logits.

    `build_model` is a zero-argument factory so each fold starts from scratch.
    The OOF logits produced here are what the CMA-ES ensemble search optimises
    on — which is why they must come from models that never saw the subject.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_logits = np.zeros((len(labels), 3), dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(sequences, labels, groups=users), start=1):
        train_ds = TensorDataset(torch.from_numpy(sequences[train_idx]),
                                 torch.from_numpy(labels[train_idx]))
        val_ds = TensorDataset(torch.from_numpy(sequences[val_idx]),
                               torch.from_numpy(labels[val_idx]))

        model = build_model().to(device)
        model, _ = fit(model,
                       make_loader(train_ds, batch_size, True, True, device),
                       make_loader(val_ds, batch_size, False, False, device),
                       device=device, **fit_kwargs)

        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(sequences[val_idx]).to(device).float()
            oof_logits[val_idx] = model(xb).cpu().numpy()

        score = f1_score(labels[val_idx], oof_logits[val_idx].argmax(1),
                         average="weighted")
        fold_scores.append(score)
        print(f"[Fold {fold}] window-level weighted F1 = {score:.4f}")

    print(f"Mean fold F1 = {np.mean(fold_scores):.4f} "
          f"(+/- {np.std(fold_scores):.4f})")
    return oof_logits, fold_scores
