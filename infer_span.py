"""
Ensemble Inference for 5-Fold Cross-Validation SCN Models

STRATEGY:
1. Load all 5 trained fold models
2. For each sample, compute probabilities from all models
3. Average the sigmoid probabilities (NOT binary predictions)
4. Apply threshold to averaged probabilities
5. Decode to spans

WHY AVERAGING PROBABILITIES WORKS:
- Averaging logits smooths the decision boundary
- Cancels random noise/hallucinations from individual models
- Reinforces strong signals where models agree
- More robust than voting on binary predictions

EXPECTED IMPACT:
- All metrics improve (stability + generalization)
- Reduces variance from individual model quirks
- Standard Kaggle winning strategy
"""

import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from transformers import DebertaV2TokenizerFast, DebertaV2Model
from collections import defaultdict
from tqdm import tqdm
import numpy as np


# ============================================================================
# Configuration
# ============================================================================
ENSEMBLE_DIR = "/scratch/ishaan.karan/scn-deberta-large-ensemble"
DEV_FILE = "dev_public.jsonl"
TEST_FILE = "test_rehydrated.jsonl"
SUBMISSION_FILE = "submission_ensemble.jsonl"
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256

MARKERS = ["Action", "Actor", "Effect", "Evidence", "Victim"]
NUM_MARKERS = len(MARKERS)
MARKER_TO_ID = {m: i for i, m in enumerate(MARKERS)}

# Per-marker span width priors (from data statistics)
MARKER_MAX_SPAN_WIDTH = {
    "Action": 15,
    "Actor": 8,
    "Effect": 18,
    "Evidence": 18,
    "Victim": 8
}

# Default thresholds (will be tuned on dev)
DEFAULT_THRESHOLDS = {
    "Action": 0.35,
    "Actor": 0.40,
    "Effect": 0.30,
    "Evidence": 0.28,
    "Victim": 0.30
}


# ============================================================================
# OFFICIAL EVALUATION: OVERLAP-BASED MACRO F1
# ============================================================================
class OverlapMacroF1Evaluator:
    """Official evaluation metric for the task."""
    
    def __init__(self, marker_types):
        self.marker_types = marker_types
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.gold = []
    
    def add(self, pred_markers, gold_markers):
        self.predictions.append(pred_markers)
        self.gold.append(gold_markers)
    
    def spans_overlap(self, span1, span2):
        start1, end1 = span1["startIndex"], span1["endIndex"]
        start2, end2 = span2["startIndex"], span2["endIndex"]
        return start1 < end2 and start2 < end1
    
    def compute_iou(self, span1, span2):
        start1, end1 = span1["startIndex"], span1["endIndex"]
        start2, end2 = span2["startIndex"], span2["endIndex"]
        
        inter_start = max(start1, start2)
        inter_end = min(end1, end2)
        
        if inter_end <= inter_start:
            return 0.0
        
        intersection = inter_end - inter_start
        union = (end1 - start1) + (end2 - start2) - intersection
        return intersection / union if union > 0 else 0.0
    
    def compute_overlap_f1_per_type(self, marker_type):
        tp, fp, fn = 0, 0, 0
        
        for pred_markers, gold_markers in zip(self.predictions, self.gold):
            pred_type = [m for m in pred_markers if m["type"] == marker_type]
            gold_type = [m for m in gold_markers if m["type"] == marker_type]
            
            pred_matched = [False] * len(pred_type)
            gold_matched = [False] * len(gold_type)
            
            matches = []
            for pi, pred in enumerate(pred_type):
                for gi, gold in enumerate(gold_type):
                    if self.spans_overlap(pred, gold):
                        iou = self.compute_iou(pred, gold)
                        matches.append((iou, pi, gi))
            
            matches.sort(reverse=True, key=lambda x: x[0])
            
            for iou, pi, gi in matches:
                if not pred_matched[pi] and not gold_matched[gi]:
                    pred_matched[pi] = True
                    gold_matched[gi] = True
                    tp += 1
            
            fp += sum(1 for m in pred_matched if not m)
            fn += sum(1 for m in gold_matched if not m)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn
        }
    
    def compute_macro_f1(self):
        per_type_metrics = {}
        f1_scores = []
        
        for marker_type in self.marker_types:
            metrics = self.compute_overlap_f1_per_type(marker_type)
            per_type_metrics[marker_type] = metrics
            f1_scores.append(metrics["f1"])
        
        macro_f1 = np.mean(f1_scores)
        macro_precision = np.mean([m["precision"] for m in per_type_metrics.values()])
        macro_recall = np.mean([m["recall"] for m in per_type_metrics.values()])
        
        return {
            "macro_f1": macro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "per_type": per_type_metrics
        }
    
    def print_report(self):
        metrics = self.compute_macro_f1()
        
        print("\n" + "="*70)
        print("      OFFICIAL EVALUATION: OVERLAP-BASED MACRO F1 (ENSEMBLE)")
        print("="*70)
        
        print("\n" + "★"*40)
        print(f"★  MACRO F1 (OFFICIAL METRIC): {metrics['macro_f1']:.4f}  ★")
        print("★"*40)
        
        print(f"\nMacro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        
        print("\n" + "-"*60)
        print(f"{'Type':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-"*60)
        
        for marker_type in self.marker_types:
            m = metrics["per_type"][marker_type]
            print(f"{marker_type:<12} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
        
        print("-"*60)
        
        return metrics


# ============================================================================
# Model (must match training)
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
        
        masked_scores = token_scores.clone()
        masked_scores[attention_mask == 0] = -1e9
        
        boost = torch.full((B, N), -1e9, device=device)
        width_weights = torch.softmax(self.width_logits, dim=0)
        
        for width in range(2, min(self.max_span_width + 1, N + 1)):
            if width <= N:
                windows = masked_scores.unfold(1, width, 1)
                span_mins, _ = windows.min(dim=2)
                weighted_mins = span_mins * width_weights[width - 2]
                
                for offset in range(width):
                    start_idx = offset
                    end_idx = N - width + offset + 1
                    if end_idx > start_idx:
                        boost[:, start_idx:end_idx] = torch.maximum(
                            boost[:, start_idx:end_idx], weighted_mins
                        )
        
        boost = torch.maximum(boost, masked_scores)
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
# Ensemble Model Wrapper
# ============================================================================
class EnsembleModel:
    """
    Wrapper that loads multiple fold models and averages their predictions.
    
    Averaging Strategy:
    - Convert logits to probabilities via sigmoid
    - Average probabilities across all models
    - This is more robust than averaging logits or voting
    """
    
    def __init__(self, model_paths, model_name, num_markers, marker_max_widths, gamma, device):
        self.models = []
        self.device = device
        
        print(f"\nLoading {len(model_paths)} ensemble models...")
        
        for i, path in enumerate(model_paths):
            print(f"  Loading model {i+1}: {path}")
            
            model = SpanConsistencyNetwork(
                model_name,
                num_markers,
                marker_max_widths=marker_max_widths,
                gamma=gamma
            )
            
            ckpt = torch.load(path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def predict_probs(self, input_ids, attention_mask):
        """
        Get averaged probabilities from all models.
        
        Returns:
            probs: [B, N, M] averaged sigmoid probabilities
        """
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
        
        # Stack and average: [num_models, B, N, M] -> [B, N, M]
        stacked = torch.stack(all_probs, dim=0)
        averaged = stacked.mean(dim=0)
        
        return averaged
    
    def eval(self):
        for model in self.models:
            model.eval()


# ============================================================================
# Decoding
# ============================================================================
def decode_to_spans(probs, offset_mapping, text, thresholds):
    """Convert token probabilities to character-level spans."""
    spans = []
    
    for m_idx, marker in enumerate(MARKERS):
        τ = thresholds.get(marker, 0.3)
        
        if torch.is_tensor(probs):
            preds = (probs[:, m_idx] >= τ).cpu().numpy()
        else:
            preds = probs[:, m_idx] >= τ
        
        # Group consecutive positive tokens
        start_tok = None
        for i in range(len(preds)):
            off = offset_mapping[i]
            if off is None or (off[0] == 0 and off[1] == 0):
                if start_tok is not None:
                    spans.append((start_tok, i - 1, marker))
                    start_tok = None
                continue
            
            if preds[i]:
                if start_tok is None:
                    start_tok = i
            else:
                if start_tok is not None:
                    spans.append((start_tok, i - 1, marker))
                    start_tok = None
        
        if start_tok is not None:
            spans.append((start_tok, len(preds) - 1, marker))
    
    # Convert to character spans
    char_spans = []
    for start_tok, end_tok, marker in spans:
        start_char, end_char = None, None
        for t in range(start_tok, end_tok + 1):
            off = offset_mapping[t]
            if off and not (off[0] == 0 and off[1] == 0):
                if start_char is None:
                    start_char = off[0]
                end_char = off[1]
        
        if start_char is not None and end_char is not None:
            char_spans.append({
                "startIndex": start_char,
                "endIndex": end_char,
                "type": marker,
                "text": text[start_char:end_char]
            })
    
    return char_spans


# ============================================================================
# Threshold Tuning
# ============================================================================
def compute_overlap_f1(pred_spans, gold_spans, marker):
    """Compute overlap-based F1 for one marker."""
    pred = [(s["startIndex"], s["endIndex"]) for s in pred_spans if s["type"] == marker]
    gold = [(g["startIndex"], g["endIndex"]) for g in gold_spans if g.get("type") == marker]
    
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    
    def iou(p, g):
        inter = max(0, min(p[1], g[1]) - max(p[0], g[0]))
        union = max(p[1], g[1]) - min(p[0], g[0])
        return inter / union if union > 0 else 0
    
    prec = sum(max(iou(p, g) for g in gold) for p in pred) / len(pred)
    rec = sum(max(iou(p, g) for p in pred) for g in gold) / len(gold)
    
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def tune_thresholds_ensemble(ensemble, data, tokenizer, device):
    """Grid search for optimal per-marker thresholds on dev set."""
    print("\n" + "="*60)
    print("THRESHOLD TUNING ON DEV SET (ENSEMBLE)")
    print("="*60)
    
    # Collect averaged probabilities for all samples
    all_probs = []
    all_gold = []
    all_offsets = []
    all_texts = []
    
    ensemble.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Computing ensemble probabilities"):
            enc = tokenizer(
                item["text"], truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_offsets_mapping=True, return_tensors="pt"
            )
            
            # Get averaged probabilities from ensemble
            probs = ensemble.predict_probs(
                enc["input_ids"].to(device), 
                enc["attention_mask"].to(device)
            )
            probs = probs.squeeze(0).cpu().numpy()
            
            all_probs.append(probs)
            all_gold.append(item.get("markers") or [])
            all_offsets.append(enc["offset_mapping"].squeeze(0).tolist())
            all_texts.append(item["text"])
    
    # Grid search for each marker independently
    threshold_range = [0.15, 0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50]
    best_thresholds = {}
    
    for marker in MARKERS:
        best_f1 = -1
        best_t = 0.3
        m_idx = MARKER_TO_ID[marker]
        
        for t in threshold_range:
            f1_scores = []
            
            for probs, gold, offsets, text in zip(all_probs, all_gold, all_offsets, all_texts):
                temp_thresh = {m: 0.5 for m in MARKERS}
                temp_thresh[marker] = t
                
                pred_spans = decode_to_spans(probs, offsets, text, temp_thresh)
                f1 = compute_overlap_f1(pred_spans, gold, marker)
                f1_scores.append(f1)
            
            avg_f1 = np.mean(f1_scores)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_t = t
        
        best_thresholds[marker] = best_t
        print(f"  {marker}: threshold = {best_t:.2f}, F1 = {best_f1:.4f}")
    
    return best_thresholds, all_probs, all_gold, all_offsets, all_texts


# ============================================================================
# Utilities
# ============================================================================
def load_data(path):
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                item.setdefault("_id", f"sample_{i}")
                item.setdefault("text", "")
                if item.get("markers") is None:
                    item["markers"] = []
                item.setdefault("conspiracy", "No")
                data.append(item)
            except:
                pass
    return data


def find_ensemble_models(ensemble_dir):
    """Find all fold model paths in the ensemble directory."""
    model_paths = []
    
    # Look for fold directories
    fold_dirs = sorted(glob.glob(os.path.join(ensemble_dir, "fold-*")))
    
    for fold_dir in fold_dirs:
        model_path = os.path.join(fold_dir, "model.pt")
        if os.path.exists(model_path):
            model_paths.append(model_path)
    
    return model_paths


def load_ensemble_config(ensemble_dir):
    """Load ensemble configuration."""
    config_path = os.path.join(ensemble_dir, "ensemble_config.json")
    
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    
    # Default config
    return {
        "model_name": MODEL_NAME,
        "markers": MARKERS,
        "marker_max_span_width": MARKER_MAX_SPAN_WIDTH,
        "gamma": 1.5
    }


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("5-FOLD ENSEMBLE INFERENCE")
    print("="*70)
    print(f"Ensemble dir: {ENSEMBLE_DIR}")
    print(f"Device: {device}")
    
    # Find ensemble models
    model_paths = find_ensemble_models(ENSEMBLE_DIR)
    
    if not model_paths:
        print(f"ERROR: No fold models found in {ENSEMBLE_DIR}")
        print("Expected structure: {ENSEMBLE_DIR}/fold-*/model.pt")
        sys.exit(1)
    
    print(f"\nFound {len(model_paths)} fold models:")
    for p in model_paths:
        print(f"  {p}")
    
    # Load config
    config = load_ensemble_config(ENSEMBLE_DIR)
    model_name = config.get("model_name", MODEL_NAME)
    marker_widths = config.get("marker_max_span_width", MARKER_MAX_SPAN_WIDTH)
    gamma = config.get("gamma", 1.5)
    
    # Create ensemble
    ensemble = EnsembleModel(
        model_paths,
        model_name,
        NUM_MARKERS,
        marker_widths,
        gamma,
        device
    )
    
    # Load tokenizer - handle fast tokenizer conversion issues
    try:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
    except Exception as e:
        print(f"  Fast tokenizer failed: {e}")
        print(f"  Falling back to slow tokenizer...")
        from transformers import DebertaV2Tokenizer
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    
    # =========================================================================
    # STEP 1: Load dev set, tune thresholds, evaluate
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Loading DEV set: {DEV_FILE}")
    print(f"{'='*70}")
    
    dev_data = load_data(DEV_FILE)
    print(f"Loaded {len(dev_data)} dev samples")
    
    # Tune thresholds on dev (using ensemble predictions)
    thresholds, dev_probs, dev_gold, dev_offsets, dev_texts = tune_thresholds_ensemble(
        ensemble, dev_data, tokenizer, device
    )
    print(f"\nTuned thresholds: {thresholds}")
    
    # Evaluate on dev with tuned thresholds
    print("\n" + "="*60)
    print("EVALUATING ON DEV SET (ENSEMBLE)")
    print("="*60)
    
    dev_predictions = []
    for probs, offsets, text in zip(dev_probs, dev_offsets, dev_texts):
        spans = decode_to_spans(probs, offsets, text, thresholds)
        dev_predictions.append(spans)
    
    evaluator = OverlapMacroF1Evaluator(MARKERS)
    for i in range(len(dev_data)):
        evaluator.add(dev_predictions[i], dev_gold[i])
    
    metrics = evaluator.print_report()
    
    # Save metrics
    metrics_file = "evaluation_metrics_ensemble.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "official_macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "thresholds": thresholds,
            "num_models": len(model_paths),
            "per_type": {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv 
                            for kk, vv in v.items()} 
                        for k, v in metrics["per_type"].items()}
        }, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")
    
    # =========================================================================
    # STEP 2: Load test set, generate predictions with ensemble
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Loading TEST set: {TEST_FILE}")
    print(f"{'='*70}")
    
    test_data = load_data(TEST_FILE)
    print(f"Loaded {len(test_data)} test samples")
    
    print("\nGenerating ensemble predictions on test set...")
    test_predictions = []
    
    ensemble.eval()
    with torch.no_grad():
        for item in tqdm(test_data, desc="Predicting (ensemble)"):
            enc = tokenizer(
                item["text"], truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_offsets_mapping=True, return_tensors="pt"
            )
            
            # Get averaged probabilities from ensemble
            probs = ensemble.predict_probs(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device)
            )
            probs = probs.squeeze(0)
            
            spans = decode_to_spans(probs, enc["offset_mapping"].squeeze(0).tolist(),
                                   item["text"], thresholds)
            test_predictions.append(spans)
    
    # Save submission file for test
    print(f"\nSaving ensemble test predictions to {SUBMISSION_FILE}...")
    with open(SUBMISSION_FILE, "w") as f:
        for item, preds in zip(test_data, test_predictions):
            f.write(json.dumps({
                "_id": item["_id"],
                "conspiracy": item["conspiracy"],
                "markers": preds
            }) + "\n")
    
    # Statistics
    total = sum(len(p) for p in test_predictions)
    by_type = defaultdict(int)
    for preds in test_predictions:
        for s in preds:
            by_type[s["type"]] += 1
    
    print(f"\nTest Prediction Statistics (Ensemble):")
    print(f"  Total spans: {total}")
    print(f"  Avg per sample: {total/len(test_data):.2f}")
    print(f"\nBy marker:")
    for m in MARKERS:
        print(f"  {m}: {by_type[m]}")
    
    print("\n" + "="*70)
    print(f"ENSEMBLE INFERENCE COMPLETE!")
    print(f"{'='*70}")
    print(f"  Models used: {len(model_paths)}")
    print(f"  Dev metrics: {metrics_file}")
    print(f"  Test submission: {SUBMISSION_FILE}")
    print("="*70)
