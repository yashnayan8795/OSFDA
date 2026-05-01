"""
Problem B — Incident Category Classification Model
====================================================
Three-tier approach following the implementation plan:

  Tier 1 — Baseline:
    TF-IDF (unigrams + bigrams, max 10k) + Logistic Regression (OVR)

  Tier 2 — Text Tower:
    Sentence-BERT embeddings of narrative + Logistic Regression (ablation)

  Tier 3 — Fusion:
    Text Tower (Sentence-BERT) + Tabular Tower (LightGBM leaf embeddings)
    → Fusion MLP → Per-label sigmoid heads → Multi-label BCE loss

All thresholds are tuned per-label on the validation set.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — TF-IDF Baseline
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_baseline(
    X_train_text: pd.Series,
    y_train: pd.DataFrame,
    X_val_text: pd.Series,
    y_val: pd.DataFrame,
    label_names: List[str],
    max_features: int = 10_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict[str, Any]:
    """
    TF-IDF + One-vs-Rest Logistic Regression baseline.

    Parameters
    ----------
    X_train_text, X_val_text : pd.Series
        Combined narrative text (cleaned).
    y_train, y_val : pd.DataFrame
        Binary indicator matrices (n_samples × n_labels).
    label_names : list of str
        Ordered category column names.

    Returns
    -------
    dict with keys: vectorizer, classifiers, thresholds, label_names
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    print("  [Baseline] Fitting TF-IDF vectorizer...")
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    X_train_tfidf = vec.fit_transform(X_train_text.fillna(""))
    X_val_tfidf = vec.transform(X_val_text.fillna(""))

    classifiers = {}
    val_probs = {}
    for label in label_names:
        y_tr = y_train[label].values
        y_va = y_val[label].values
        # Class weight for imbalance
        pos_weight = max(1, (y_tr == 0).sum() / max(y_tr.sum(), 1))
        clf = LogisticRegression(
            C=1.0,
            class_weight={0: 1.0, 1: pos_weight},
            max_iter=1000,
            solver="saga",
            n_jobs=-1,
        )
        clf.fit(X_train_tfidf, y_tr)
        classifiers[label] = clf
        val_probs[label] = clf.predict_proba(X_val_tfidf)[:, 1]

    # Per-label threshold tuning on validation
    thresholds = tune_thresholds(val_probs, y_val, label_names)

    return {
        "type": "tfidf_baseline",
        "vectorizer": vec,
        "classifiers": classifiers,
        "thresholds": thresholds,
        "label_names": label_names,
    }


def predict_tfidf(
    model: Dict[str, Any],
    X_text: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (probs, binary_preds) arrays of shape (n_samples, n_labels).
    """
    X_tfidf = model["vectorizer"].transform(X_text.fillna(""))
    label_names = model["label_names"]
    probs = np.column_stack([
        model["classifiers"][label].predict_proba(X_tfidf)[:, 1]
        for label in label_names
    ])
    preds = (probs >= np.array([model["thresholds"][l] for l in label_names])).astype(int)
    return probs, preds


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Sentence-BERT Text Tower
# ─────────────────────────────────────────────────────────────────────────────

def encode_texts(
    texts: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_seq_length: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode texts with Sentence-BERT. Returns (n_samples, embedding_dim) array.
    Truncates to max_seq_length to control memory/speed.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  [TextTower] Loading model: {model_name}")
    encoder = SentenceTransformer(model_name)
    encoder.max_seq_length = max_seq_length

    texts_clean = texts.fillna("").tolist()
    print(f"  [TextTower] Encoding {len(texts_clean)} texts (batch={batch_size})...")
    embeddings = encoder.encode(
        texts_clean,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return embeddings


def build_text_tower(
    embeddings_train: np.ndarray,
    y_train: pd.DataFrame,
    embeddings_val: np.ndarray,
    y_val: pd.DataFrame,
    label_names: List[str],
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Logistic Regression on top of Sentence-BERT embeddings (ablation: text only).
    """
    from sklearn.linear_model import LogisticRegression

    classifiers = {}
    val_probs = {}
    for label in label_names:
        y_tr = y_train[label].values
        pos_weight = max(1, (y_tr == 0).sum() / max(y_tr.sum(), 1))
        clf = LogisticRegression(
            C=C,
            class_weight={0: 1.0, 1: pos_weight},
            max_iter=1000,
            solver="lbfgs",
        )
        clf.fit(embeddings_train, y_tr)
        classifiers[label] = clf
        val_probs[label] = clf.predict_proba(embeddings_val)[:, 1]

    thresholds = tune_thresholds(val_probs, y_val, label_names)

    return {
        "type": "text_tower",
        "classifiers": classifiers,
        "thresholds": thresholds,
        "label_names": label_names,
    }


def predict_text_tower(
    model: Dict[str, Any],
    embeddings: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    label_names = model["label_names"]
    probs = np.column_stack([
        model["classifiers"][label].predict_proba(embeddings)[:, 1]
        for label in label_names
    ])
    preds = (probs >= np.array([model["thresholds"][l] for l in label_names])).astype(int)
    return probs, preds


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Fusion Model (Text + Tabular)
# ─────────────────────────────────────────────────────────────────────────────

def get_lgbm_leaf_embeddings(
    lgbm_model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Extract leaf index embeddings from a trained LightGBM model.
    Each tree produces a leaf index; concatenated they form a sparse
    categorical representation of the tabular input.
    """
    leaves = lgbm_model.predict(X, pred_leaf=True)
    return leaves.astype(np.float32)


class FusionMLP:
    """
    Two-tower fusion: Sentence-BERT embeddings + tabular features
    → MLP → Multi-label BCE loss.
    Implemented in PyTorch.
    """
    def __init__(
        self,
        hidden_dims: Tuple[int, int] = (256, 128),
        lr: float = 5e-4,
        max_epochs: int = 30,
        batch_size: int = 128,
        patience: int = 3,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.thresholds: Dict[str, float] = {}
        self.label_names: List[str] = []

    def fit(
        self,
        text_train: np.ndarray,
        tabular_train: np.ndarray,
        y_train: pd.DataFrame,
        text_val: np.ndarray,
        tabular_val: np.ndarray,
        y_val: pd.DataFrame,
        label_names: List[str],
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        self.label_names = label_names
        
        X_tr_raw = np.hstack([text_train, tabular_train]).astype(np.float32)
        X_va_raw = np.hstack([text_val, tabular_val]).astype(np.float32)
        
        self.scaler = StandardScaler()
        X_tr = self.scaler.fit_transform(X_tr_raw)
        X_va = self.scaler.transform(X_va_raw)
        
        Y_tr = y_train[label_names].values.astype(np.float32)
        Y_va = y_val[label_names].values.astype(np.float32)
        
        input_dim = X_tr.shape[1]
        num_classes = len(label_names)

        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr))
        val_ds = TensorDataset(torch.tensor(X_va), torch.tensor(Y_va))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dims[1], num_classes)
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            val_loss /= len(val_ds)
            print(f"    Epoch {epoch+1}/{self.max_epochs} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("    Early stopping.")
                    break
                    
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        # Tune thresholds
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(torch.tensor(X_va))
            val_probs = torch.sigmoid(val_logits).numpy()
            
        prob_dict = {l: val_probs[:, i] for i, l in enumerate(label_names)}
        self.thresholds = tune_thresholds(prob_dict, y_val, label_names)

    def predict(
        self,
        text_embeddings: np.ndarray,
        tabular_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        self.model.eval()
        X_test_raw = np.hstack([text_embeddings, tabular_features]).astype(np.float32)
        X_test = self.scaler.transform(X_test_raw)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_test))
            probs = torch.sigmoid(logits).numpy()
            
        thresholds = np.array([self.thresholds[l] for l in self.label_names])
        preds = (probs >= thresholds).astype(int)
        return probs, preds


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_thresholds(
    val_probs: Dict[str, np.ndarray],
    y_val: pd.DataFrame,
    label_names: List[str],
    thresholds_to_try: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Find the per-label probability threshold that maximizes F1 on validation.

    Returns
    -------
    dict mapping label → optimal threshold
    """
    from sklearn.metrics import f1_score

    if thresholds_to_try is None:
        thresholds_to_try = np.arange(0.05, 0.95, 0.05)

    best_thresholds = {}
    for label in label_names:
        probs = val_probs[label]
        y_true = y_val[label].values
        if y_true.sum() == 0:
            logger.warning(
                "Label '%s' has no positive examples in validation set — "
                "defaulting threshold to 0.5. Model cannot tune this label.",
                label,
            )
            best_thresholds[label] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in thresholds_to_try:
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[label] = float(best_t)

    return best_thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Model Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_category_model(model: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved model to {path}")
    return path


def load_category_model(path: Path) -> Any:
    return joblib.load(path)
