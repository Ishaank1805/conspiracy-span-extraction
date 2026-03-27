"""
5-Fold Cross-Validation Ensemble Training for SCN

STRATEGY:
1. Split train_rehydrated.jsonl into 5 stratified folds
2. Train 5 separate SCN models (each on 4/5 of data)
3. Save all 5 models for ensemble inference

WHY IT WORKS:
- Deep learning on small/medium datasets has high variance
- One seed might excel at "Actor" but fail at "Evidence"
- Averaging logits smooths decision boundaries
- Cancels random noise, reinforces strong signals

EXPECTED IMPACT:
- All metrics improve (stability + generalization)
- Standard Kaggle winning strategy
- 5x training compute (worth it for competitions)

USAGE:
    # Train all folds sequentially (original behavior)
    python train_span_kfold.py
    
    # Train specific fold (for parallel training)
    python train_span_kfold.py --fold 0
    python train_span_kfold.py --fold 1
    python train_span_kfold.py --fold 2
    python train_span_kfold.py --fold 3
    python train_span_kfold.py --fold 4
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import DebertaV2TokenizerFast, DebertaV2Model
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import math
import numpy as np
from collections import Counter
import argparse
import sys

# ============================================================================
# Configuration
# ============================================================================
TRAIN_FILE = "train_rehydrated.jsonl"
MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = "/scratch/ishaan.karan/scn-deberta-large-ensemble"

# Ensemble config
NUM_FOLDS = 5
RANDOM_SEED = 42

# Training params (per fold)
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 12
WARMUP_RATIO = 0.1

# Sequence params
MAX_LENGTH = 128

# Per-marker span width priors
MARKER_MAX_SPAN_WIDTH = {
    "Action": 12,
    "Actor": 6,
    "Effect": 14,
    "Evidence": 14,
    "Victim": 6
}

# Tversky Loss params (RECALL-BIASED)
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
LAMBDA_SCRD = 0.15
GAMMA = 1.5

# Markers
MARKERS = ["Action", "Actor", "Effect", "Evidence", "Victim"]
NUM_MARKERS = len(MARKERS)
MARKER_TO_ID = {m: i for i, m in enumerate(MARKERS)}

# Class weights
MANUAL_CLASS_WEIGHTS = {
    "Action": 1.0,
    "Actor": 0.85,
    "Effect": 1.10,
    "Evidence": 1.12,
    "Victim": 1.20
}

# Memory optimization
USE_FP16 = True
GRADIENT_CHECKPOINTING = False


# ============================================================================
# Model Components (same as train_span_tversky.py)
# ============================================================================
class SpanConsistencyLayer(nn.Module):
    def __init__(self, max_span_width=15, gamma=1.0):
        super().__init__()
        self.max_span_width = max_span_width
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.width_logits = nn.Parameter(torch.zeros(max_span_width))
    
    def forward(self, token_scores, attention_mask):
        B, N = token_scores.shape
        device = token_scores.device
        
        NEG_INF = -1e4
        masked_scores = token_scores.masked_fill(attention_mask == 0, NEG_INF)
        width_weights = torch.softmax(self.width_logits, dim=0)
        
        all_boosts = [masked_scores]
        
        for width in range(2, min(self.max_span_width + 1, N + 1)):
            windows = masked_scores.unfold(1, width, 1)
            span_mins, _ = windows.min(dim=2)
            
            w_idx = min(width - 2, len(width_weights) - 1)
            weighted_mins = span_mins * width_weights[w_idx]
            
            shifts = []
            num_spans = span_mins.size(1)
            
            for offset in range(width):
                left_pad = offset
                right_pad = N - num_spans - offset
                if right_pad >= 0:
                    shifted = F.pad(weighted_mins, (left_pad, right_pad), value=NEG_INF)
                    shifts.append(shifted)
            
            if shifts:
                stacked = torch.stack(shifts, dim=0)
                expanded = stacked.max(dim=0)[0]
                all_boosts.append(expanded)
        
        all_boosts_stack = torch.stack(all_boosts, dim=0)
        boost = all_boosts_stack.max(dim=0)[0]
        
        return token_scores + self.gamma * boost


class CrossMarkerAttention(nn.Module):
    def __init__(self, num_markers, hidden_dim=64):
        super().__init__()
        self.num_markers = num_markers
        self.correlation_matrix = nn.Parameter(torch.eye(num_markers) * 0.5)
        self.cross_mlp = nn.Sequential(
            nn.Linear(num_markers, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_markers)
        )
        self.gate = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, marker_logits):
        probs = torch.sigmoid(marker_logits)
        correlated = torch.matmul(probs, self.correlation_matrix)
        refined = self.cross_mlp(correlated)
        output = marker_logits + torch.sigmoid(self.gate) * refined
        return output


class SpanConsistencyNetwork(nn.Module):
    def __init__(self, model_name, num_markers, marker_max_widths, gamma=1.0):
        super().__init__()
        self.encoder = DebertaV2Model.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.num_markers = num_markers
        self.marker_max_widths = marker_max_widths
        
        self.token_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
            for _ in range(num_markers)
        ])
        
        self.consistency_layers = nn.ModuleList([
            SpanConsistencyLayer(
                max_span_width=marker_max_widths[MARKERS[m]],
                gamma=gamma
            )
            for m in range(num_markers)
        ])
        
        self.cross_marker_attn = CrossMarkerAttention(num_markers, hidden_dim=64)
    
    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        marker_logits = []
        for m in range(self.num_markers):
            token_scores = self.token_scorers[m](h).squeeze(-1)
            boosted_scores = self.consistency_layers[m](token_scores, attention_mask)
            marker_logits.append(boosted_scores)
        
        stacked_logits = torch.stack(marker_logits, dim=-1)
        refined_logits = self.cross_marker_attn(stacked_logits)
        
        return refined_logits


# ============================================================================
# Loss (Tversky + Span Count Regularization)
# ============================================================================
class SCRTLoss(nn.Module):
    """Span-Count Regularized Tversky Loss (SCRT) - RECALL BIASED"""
    
    def __init__(self, alpha=0.3, beta=0.7, lambda_reg=0.1, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.smooth = smooth
    
    def forward(self, logits, targets, mask, class_weights):
        probs = torch.sigmoid(logits)
        mask_f = mask.float()
        
        total_loss = 0.0
        
        for m in range(probs.shape[-1]):
            p_m = probs[:, :, m] * mask_f
            y_m = targets[:, :, m] * mask_f
            
            # Tversky components
            TP = (p_m * y_m).sum()
            FP = (p_m * (1 - y_m)).sum()
            FN = ((1 - p_m) * y_m).sum()
            
            tversky = (TP + self.smooth) / (
                TP + self.alpha * FP + self.beta * FN + self.smooth
            )
            tversky_loss = 1.0 - tversky
            
            # Span count regularization
            trans_p = torch.abs(p_m[:, 1:] - p_m[:, :-1])
            trans_mask = mask_f[:, 1:] * mask_f[:, :-1]
            B_pred = (trans_p * trans_mask).sum(dim=1) / 2.0
            
            y_padded = torch.cat([torch.zeros_like(y_m[:, :1]), y_m], dim=1)
            starts = (y_padded[:, 1:] > y_padded[:, :-1]).float()
            B_gold = (starts * mask_f).sum(dim=1)
            
            span_loss = ((B_pred - B_gold) ** 2).mean()
            
            total_loss += class_weights[m] * (tversky_loss + self.lambda_reg * span_loss)
        
        return total_loss / class_weights.sum()


# ============================================================================
# Data Processing
# ============================================================================
def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data


def compute_class_weights(data):
    weights = MANUAL_CLASS_WEIGHTS.copy()
    total = sum(weights.values())
    weights = {m: w * NUM_MARKERS / total for m, w in weights.items()}
    return torch.tensor([weights[m] for m in MARKERS])


def create_labels(markers, offset_mapping):
    labels = torch.zeros(len(offset_mapping), NUM_MARKERS)
    
    for marker in markers:
        mtype = marker.get("type")
        if mtype not in MARKER_TO_ID:
            continue
        
        m_idx = MARKER_TO_ID[mtype]
        start_char, end_char = marker.get("startIndex", 0), marker.get("endIndex", 0)
        
        for tok_idx, (ts, te) in enumerate(offset_mapping):
            if ts is None or te is None or (ts == 0 and te == 0):
                continue
            if ts < end_char and te > start_char:
                labels[tok_idx, m_idx] = 1.0
    
    return labels


class SpanDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item.get("text", ""), truncation=True, padding="max_length",
            max_length=self.max_length, return_offsets_mapping=True, return_tensors="pt"
        )
        labels = create_labels(item.get("markers", []), enc["offset_mapping"].squeeze(0).tolist())
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch])
    }


# ============================================================================
# Stratification Helper
# ============================================================================
def get_stratification_labels(data):
    """
    Create stratification labels based on marker presence.
    Each sample gets a label based on which markers are present.
    """
    labels = []
    for item in data:
        markers_present = set()
        for marker in item.get("markers", []):
            mtype = marker.get("type")
            if mtype in MARKERS:
                markers_present.add(mtype)
        
        # Create a binary encoding of markers present
        label = 0
        for i, m in enumerate(MARKERS):
            if m in markers_present:
                label += (1 << i)
        labels.append(label)
    
    return np.array(labels)


def create_folds(data, num_folds, seed):
    """
    Create stratified folds based on marker presence.
    Falls back to simple KFold if stratification fails.
    """
    labels = get_stratification_labels(data)
    
    # Count label frequencies
    label_counts = Counter(labels)
    
    # Check if stratification is possible
    # (each label must appear at least num_folds times)
    can_stratify = all(count >= num_folds for count in label_counts.values())
    
    if can_stratify:
        print(f"Using stratified K-fold (seed={seed})")
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = list(kfold.split(np.zeros(len(data)), labels))
    else:
        print(f"Using simple K-fold (some labels too rare for stratification)")
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = list(kfold.split(np.zeros(len(data))))
    
    return folds


# ============================================================================
# Training
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scheduler, device, class_weights, grad_accum, scaler=None):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels, attention_mask, class_weights.to(device))
                loss = loss / grad_accum
            
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels, attention_mask, class_weights.to(device))
            loss = loss / grad_accum
            
            loss.backward()
            
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
    
    return total_loss / len(loader)


def train_fold(fold_idx, train_indices, val_indices, full_dataset, tokenizer, device):
    """Train a single fold and return the model."""
    
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"{'='*70}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # Create data loaders
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    
    # Initialize fresh model for this fold
    model = SpanConsistencyNetwork(
        MODEL_NAME, 
        NUM_MARKERS, 
        marker_max_widths=MARKER_MAX_SPAN_WIDTH,
        gamma=GAMMA
    )
    model.to(device)
    
    # Loss
    criterion = SCRTLoss(
        alpha=TVERSKY_ALPHA, 
        beta=TVERSKY_BETA,
        lambda_reg=LAMBDA_SCRD
    )
    
    # Class weights
    class_weights = compute_class_weights(full_dataset.data)
    
    # Optimizer
    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LEARNING_RATE},
        {"params": model.token_scorers.parameters(), "lr": LEARNING_RATE * 5},
        {"params": model.consistency_layers.parameters(), "lr": LEARNING_RATE * 5},
        {"params": model.cross_marker_attn.parameters(), "lr": LEARNING_RATE * 5}
    ], weight_decay=0.01)
    
    # Scheduler
    num_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = int(num_steps * WARMUP_RATIO)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if USE_FP16 and device.type == "cuda" else None
    
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                              device, class_weights, GRAD_ACCUM, scaler)
        print(f"  Fold {fold_idx + 1} | Epoch {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")
    
    return model


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="5-Fold CV Training for SCN")
    parser.add_argument("--fold", type=int, default=None, 
                        help="Specific fold to train (0-4). If not provided, trains all folds sequentially.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("5-FOLD CROSS-VALIDATION ENSEMBLE TRAINING")
    print("="*70)
    print(f"Model: SCN + DeBERTa-v3-large + Tversky Loss")
    print(f"Folds: {NUM_FOLDS}")
    print(f"Seed: {RANDOM_SEED}")
    print(f"Device: {device}")
    
    if args.fold is not None:
        print(f"Training ONLY fold: {args.fold}")
    else:
        print(f"Training ALL folds sequentially")
    print("="*70)
    
    # Validate fold argument
    if args.fold is not None and (args.fold < 0 or args.fold >= NUM_FOLDS):
        print(f"ERROR: --fold must be between 0 and {NUM_FOLDS - 1}")
        sys.exit(1)
    
    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Load data
    print(f"\nLoading data from {TRAIN_FILE}...")
    train_data = load_data(TRAIN_FILE)
    print(f"Loaded {len(train_data)} training examples")
    
    # Tokenizer - handle fast tokenizer conversion issues
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    try:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"  Fast tokenizer failed: {e}")
        print(f"  Falling back to slow tokenizer...")
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    
    # Create full dataset
    full_dataset = SpanDataset(train_data, tokenizer, MAX_LENGTH)
    
    # Create folds (SAME SEED ensures all parallel jobs get same splits)
    print(f"\nCreating {NUM_FOLDS} folds...")
    folds = create_folds(train_data, NUM_FOLDS, RANDOM_SEED)
    
    # Determine which folds to train
    if args.fold is not None:
        folds_to_train = [(args.fold, folds[args.fold])]
    else:
        folds_to_train = list(enumerate(folds))
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train specified fold(s)
    for fold_idx, (train_indices, val_indices) in folds_to_train:
        model = train_fold(fold_idx, train_indices, val_indices, full_dataset, tokenizer, device)
        
        # Save fold model
        fold_dir = os.path.join(OUTPUT_DIR, f"fold-{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        torch.save({
            "fold": fold_idx,
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": MODEL_NAME,
                "markers": MARKERS,
                "marker_max_span_width": MARKER_MAX_SPAN_WIDTH,
                "gamma": GAMMA,
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "num_folds": NUM_FOLDS
            },
            "val_indices": val_indices.tolist() if hasattr(val_indices, 'tolist') else list(val_indices)
        }, os.path.join(fold_dir, "model.pt"))
        
        print(f"  Saved fold {fold_idx} to {fold_dir}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save tokenizer once (only if training all folds or fold 0)
    if args.fold is None or args.fold == 0:
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Save ensemble config
        with open(os.path.join(OUTPUT_DIR, "ensemble_config.json"), "w") as f:
            json.dump({
                "num_folds": NUM_FOLDS,
                "model_name": MODEL_NAME,
                "markers": MARKERS,
                "marker_max_span_width": MARKER_MAX_SPAN_WIDTH,
                "gamma": GAMMA,
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "seed": RANDOM_SEED
            }, f, indent=2)
    
    print(f"\n{'='*70}")
    if args.fold is not None:
        print(f"FOLD {args.fold} TRAINING COMPLETE!")
    else:
        print(f"ALL FOLDS TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if args.fold is not None:
        print(f"\nTo train remaining folds in parallel, run:")
        for i in range(NUM_FOLDS):
            if i != args.fold:
                print(f"  python train_span_kfold.py --fold {i}")
    
    print(f"\nAfter all folds are trained, run ensemble inference:")
    print(f"  python infer_span_ensemble.py")
    print(f"{'='*70}")
