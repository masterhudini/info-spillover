import os
import logging
from typing import Dict, List, Tuple

import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.modules.data.modeling_data_module import ModelingDataModule
from src.modules.models.static_gnn import StaticGNNClassifier, StaticGNNMultiTaskClassifier

logger = logging.getLogger(__name__)

def _build_adjacency(network, subreddits: List[str]) -> torch.Tensor:
    n = len(subreddits)
    index = {name: i for i, name in enumerate(subreddits)}
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v, data in network.edges(data=True):
        if u not in index or v not in index:
            raise ValueError(f"Network node not in selected subreddits: {u}, {v}")
        weight = float(data.get("weight", 0.0))
        adj[index[u], index[v]] = weight

    row_sums = adj.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        adj_norm = np.where(row_sums > 0, adj / row_sums, 0.0)
    return torch.tensor(adj_norm, dtype=torch.float32)


def _map_targets(y: np.ndarray) -> np.ndarray:
    mapping = {-1: 0, 0: 1, 1: 2}
    out = np.full_like(y, fill_value=-1, dtype=np.int64)
    for src, dst in mapping.items():
        out[y == src] = dst
    if (out < 0).any():
        invalid = np.unique(y[out < 0])
        raise ValueError(f"Invalid return_direction values: {invalid}")
    return out


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    f1_scores = []
    for cls in [0, 1, 2]:
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _eval_predictions(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb)
            batch_preds = torch.argmax(logits, dim=1)
            preds.append(batch_preds.cpu().numpy())
    return np.concatenate(preds)


def _validate_search_space(search_space: Dict[str, List]) -> None:
    required = ["hidden_dim", "gnn_dim", "epochs", "lr", "batch_size"]
    missing = [k for k in required if k not in search_space]
    if missing:
        raise ValueError(f"Missing optuna_search_space keys: {missing}")
    for key in required:
        if not isinstance(search_space[key], list) or not search_space[key]:
            raise ValueError(f"optuna_search_space[{key}] must be a non-empty list")


def _prepare_training_data(
    processed_data,
    network,
    feature_columns: List[str],
    dy_freq: str,
    ffill_limit: int,
    subreddits: List[str],
    symbols: List[str],
    val_split: float,
    test_split: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    data_module = ModelingDataModule(
        processed_data=processed_data,
        network=network,
        feature_columns=feature_columns,
        dy_resample_freq=dy_freq,
        ffill_limit=ffill_limit,
        subreddits=subreddits,
        symbols=symbols,
    )

    features, timestamps = data_module.build_feature_tensor()
    targets = data_module.build_targets(timestamps)
    returns = data_module.build_return_series(timestamps)
    adjacency = _build_adjacency(network, subreddits)

    num_timestamps = len(timestamps)
    num_symbols = len(symbols)
    y = np.full((num_timestamps, num_symbols), np.nan, dtype=np.float32)
    r = np.full((num_timestamps, num_symbols), np.nan, dtype=np.float32)
    for idx, symbol in enumerate(symbols):
        y[:, idx] = targets[symbol]
        r[:, idx] = returns[symbol]

    valid_returns = np.isfinite(r).all(axis=1)
    valid_targets = ~np.isnan(y).any(axis=1)
    valid_rows = valid_returns & valid_targets
    if not valid_rows.all():
        dropped = int((~valid_rows).sum())
        logger.info("Dropping %d timestamps with missing targets/returns", dropped)
        features = features[valid_rows]
        y = y[valid_rows]
        r = r[valid_rows]
        timestamps = [ts for i, ts in enumerate(timestamps) if valid_rows[i]]
        num_timestamps = len(timestamps)

    train_idx, val_idx, test_idx = data_module.time_split_indices(num_timestamps, val_split, test_split)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Insufficient split sizes for multitask training")

    return features, y, r, timestamps, train_idx, val_idx, test_idx, adjacency


def _masked_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError("logits must be (batch, tasks, classes)")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_y = y.reshape(-1)
    flat_mask = mask.reshape(-1)
    if flat_mask.sum() == 0:
        raise ValueError("No valid labels available for loss computation")
    return criterion(flat_logits[flat_mask], flat_y[flat_mask])


def _eval_multitask_metrics(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    mask_np = mask.cpu().numpy()

    per_task = {}
    scores = []
    for idx in range(y_true_np.shape[1]):
        valid = mask_np[:, idx]
        if not valid.any():
            continue
        metrics = _compute_metrics(y_true_np[valid, idx], preds[valid, idx])
        per_task[idx] = metrics
        scores.append(metrics["macro_f1"])
    avg_f1 = float(np.mean(scores)) if scores else 0.0
    return avg_f1, per_task


def _train_for_val_f1_multitask(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    adjacency: torch.Tensor,
    hidden_dim: int,
    gnn_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> float:
    mask = ~np.isnan(y)
    y_safe = np.where(mask, y, 0).astype(int)
    y_mapped = _map_targets(y_safe)

    x_train = torch.tensor(x[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_mapped[train_idx], dtype=torch.long)
    mask_train = torch.tensor(mask[train_idx], dtype=torch.bool)
    x_val = torch.tensor(x[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_mapped[val_idx], dtype=torch.long)
    mask_val = torch.tensor(mask[val_idx], dtype=torch.bool)

    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val, mask_val), batch_size=batch_size, shuffle=False)

    model = StaticGNNMultiTaskClassifier(
        num_features=x.shape[-1],
        hidden_dim=hidden_dim,
        gnn_dim=gnn_dim,
        num_classes=3,
        num_tasks=y.shape[1],
        adjacency=adjacency,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = _masked_loss(logits, yb, mb, criterion)
            loss.backward()
            optimizer.step()

    model.eval()
    all_logits = []
    all_true = []
    all_mask = []
    with torch.no_grad():
        for xb, yb, mb in val_loader:
            logits = model(xb)
            all_logits.append(logits)
            all_true.append(yb)
            all_mask.append(mb)
    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_true, dim=0)
    mask_val = torch.cat(all_mask, dim=0)
    avg_f1, _ = _eval_multitask_metrics(logits, y_true, mask_val)
    return avg_f1


def _select_hyperparams_via_optuna(
    processed_data,
    network,
    feature_columns: List[str],
    dy_freq: str,
    ffill_limit: int,
    subreddits: List[str],
    symbols: List[str],
    val_split: float,
    test_split: float,
    search_space: Dict[str, List],
    n_trials: int,
) -> Dict[str, float]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for hyperparameter search") from exc

    _validate_search_space(search_space)
    features, targets, _, _, train_idx, val_idx, _, adjacency = _prepare_training_data(
        processed_data=processed_data,
        network=network,
        feature_columns=feature_columns,
        dy_freq=dy_freq,
        ffill_limit=ffill_limit,
        subreddits=subreddits,
        symbols=symbols,
        val_split=val_split,
        test_split=test_split,
    )

    def objective(trial: "optuna.Trial") -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", search_space["hidden_dim"])
        gnn_dim = trial.suggest_categorical("gnn_dim", search_space["gnn_dim"])
        epochs = trial.suggest_categorical("epochs", search_space["epochs"])
        lr = trial.suggest_categorical("lr", search_space["lr"])
        batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])

        return _train_for_val_f1_multitask(
            x=features,
            y=targets,
            train_idx=train_idx,
            val_idx=val_idx,
            adjacency=adjacency,
            hidden_dim=int(hidden_dim),
            gnn_dim=int(gnn_dim),
            lr=float(lr),
            epochs=int(epochs),
            batch_size=int(batch_size),
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    best_payload = {
        "best_params": best,
        "best_value": float(study.best_value),
        "n_trials": int(n_trials),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    output_path = os.getenv("OPTUNA_BEST_PARAMS_PATH", "results/optuna_best_params.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_payload, f, indent=2, sort_keys=True)
    logger.info("Optuna best params saved to: %s", output_path)
    logger.info("Optuna best params: %s", best)
    return best


def run_modeling(processed_data, network, hyperparams_path: str) -> Dict[str, Dict[str, float]]:
    subreddits_env = os.getenv("SUBREDDITS_SELECTED")
    if not subreddits_env:
        raise ValueError("Missing required env var: SUBREDDITS_SELECTED")
    subreddits = [s.strip() for s in subreddits_env.split(",") if s.strip()]

    symbols_env = os.getenv("SYMBOLS_SELECTED")
    if not symbols_env:
        raise ValueError("Missing required env var: SYMBOLS_SELECTED")
    symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]

    feature_env = os.getenv("MODEL_FEATURE_COLUMNS")
    if not feature_env:
        raise ValueError("Missing required env var: MODEL_FEATURE_COLUMNS")
    feature_columns = [s.strip() for s in feature_env.split(",") if s.strip()]

    dy_freq = os.getenv("DY_RESAMPLE_FREQ")
    if not dy_freq:
        raise ValueError("Missing required env var: DY_RESAMPLE_FREQ")

    ffill_limit = int(os.getenv("DY_FFILL_LIMIT"))
    val_split = float(os.getenv("MODEL_VAL_SPLIT"))
    test_split = float(os.getenv("MODEL_TEST_SPLIT"))
    if not hyperparams_path:
        raise ValueError("Missing hyperparams_path")
    with open(hyperparams_path, "r") as f:
        hyperparams = yaml.safe_load(f) or {}

    model_cfg = hyperparams.get("model", {})
    if not model_cfg and "optuna_search_space" in hyperparams:
        trials_env = os.getenv("OPTUNA_TRIALS")
        if not trials_env:
            raise ValueError("Missing required env var: OPTUNA_TRIALS")
        model_cfg = _select_hyperparams_via_optuna(
            processed_data=processed_data,
            network=network,
            feature_columns=feature_columns,
            dy_freq=dy_freq,
            ffill_limit=ffill_limit,
            subreddits=subreddits,
            symbols=symbols,
            val_split=val_split,
            test_split=test_split,
            search_space=hyperparams["optuna_search_space"],
            n_trials=int(trials_env),
        )

    if not model_cfg:
        raise ValueError("Missing model hyperparameters")
    hidden_dim = int(model_cfg.get("hidden_dim"))
    gnn_dim = int(model_cfg.get("gnn_dim"))
    epochs = int(model_cfg.get("epochs"))
    lr = float(model_cfg.get("lr"))
    batch_size = int(model_cfg.get("batch_size"))
    features, targets, returns, timestamps, train_idx, val_idx, test_idx, adjacency = _prepare_training_data(
        processed_data=processed_data,
        network=network,
        feature_columns=feature_columns,
        dy_freq=dy_freq,
        ffill_limit=ffill_limit,
        subreddits=subreddits,
        symbols=symbols,
        val_split=val_split,
        test_split=test_split,
    )

    mask = ~np.isnan(targets)
    if not mask.all():
        bad = np.argwhere(~mask)
        sample = bad[:5].tolist()
        raise ValueError(f"Missing targets for some symbol/timestamp pairs: {sample}")
    y_safe = np.where(mask, targets, 0).astype(int)
    y_mapped = _map_targets(y_safe)

    x_train = torch.tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_mapped[train_idx], dtype=torch.long)
    mask_train = torch.tensor(mask[train_idx], dtype=torch.bool)
    x_val = torch.tensor(features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_mapped[val_idx], dtype=torch.long)
    mask_val = torch.tensor(mask[val_idx], dtype=torch.bool)
    x_test = torch.tensor(features[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y_mapped[test_idx], dtype=torch.long)
    mask_test = torch.tensor(mask[test_idx], dtype=torch.bool)

    train_loader = DataLoader(TensorDataset(x_train, y_train, mask_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val, mask_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test, mask_test), batch_size=batch_size, shuffle=False)

    logger.info(
        "Training multitask symbols=%d samples=%d train=%d val=%d test=%d features=%d",
        len(symbols),
        len(features),
        len(train_idx),
        len(val_idx),
        len(test_idx),
        features.shape[-1],
    )

    model = StaticGNNMultiTaskClassifier(
        num_features=features.shape[-1],
        hidden_dim=hidden_dim,
        gnn_dim=gnn_dim,
        num_classes=3,
        num_tasks=len(symbols),
        adjacency=adjacency,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = _masked_loss(logits, yb, mb, criterion)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        model.eval()
        all_logits = []
        all_true = []
        all_mask = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                logits = model(xb)
                all_logits.append(logits)
                all_true.append(yb)
                all_mask.append(mb)
        logits = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_true, dim=0)
        mask_val = torch.cat(all_mask, dim=0)
        avg_f1, _ = _eval_multitask_metrics(logits, y_true, mask_val)
        logger.info(
            "Multitask epoch=%d/%d loss=%.6f val_macro_f1=%.4f",
            epoch,
            epochs,
            avg_loss,
            avg_f1,
        )

    def _eval_split(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_logits = []
        all_true = []
        all_mask = []
        with torch.no_grad():
            for xb, yb, mb in loader:
                logits = model(xb)
                all_logits.append(logits)
                all_true.append(yb)
                all_mask.append(mb)
        logits = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_true, dim=0)
        mask_eval = torch.cat(all_mask, dim=0)
        preds = torch.argmax(logits, dim=-1)
        return (
            logits.cpu().numpy(),
            preds.cpu().numpy(),
            y_true.cpu().numpy(),
            mask_eval.cpu().numpy(),
        )

    train_logits, train_preds, train_true, train_mask = _eval_split(train_loader)
    val_logits, val_preds, val_true, val_mask = _eval_split(val_loader)
    test_logits, test_preds, test_true, test_mask = _eval_split(test_loader)

    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    results = {}
    test_timestamps = [str(timestamps[i]) for i in test_idx]
    for idx, symbol in enumerate(symbols):
        for label, arr in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            if not arr[:, idx].all():
                raise ValueError(f"Missing targets in {label} split for symbol: {symbol}")

        train_metrics = _compute_metrics(train_true[:, idx], train_preds[:, idx])
        val_metrics = _compute_metrics(val_true[:, idx], val_preds[:, idx])
        test_metrics = _compute_metrics(test_true[:, idx], test_preds[:, idx])

        test_returns = returns[test_idx, idx]
        if not np.isfinite(test_returns).all():
            raise ValueError(f"Non-finite returns in test split for symbol: {symbol}")

        results[symbol] = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "test_predictions": test_preds[:, idx],
            "test_probabilities": test_probs[:, idx, :].tolist(),
            "test_returns": test_returns,
            "test_timestamps": test_timestamps,
            "test_true": test_true[:, idx].tolist(),
        }

    return results
