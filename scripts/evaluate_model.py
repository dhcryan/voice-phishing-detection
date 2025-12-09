#!/usr/bin/env python
"""
Model Evaluation Script - EER, min-tDCF 등 성능 평가
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Equal Error Rate 계산"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    try:
        if len(fpr) < 2:
            return 0.5, 0.5
        eer_threshold = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
        eer = interp1d(fpr, fnr)(eer_threshold)
    except (ValueError, Exception):
        eer = 0.5
        eer_threshold = 0.5
    
    return float(eer), float(eer_threshold)


def compute_min_tdcf(
    scores: np.ndarray, 
    labels: np.ndarray,
    Pspoof: float = 0.05,
    Cmiss: float = 1,
    Cfa: float = 10
) -> float:
    """minimum tandem Detection Cost Function 계산"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    Ptar = 1 - Pspoof
    tDCF = Cmiss * fnr * Ptar + Cfa * fpr * Pspoof
    min_tDCF = np.min(tDCF)
    
    return float(min_tDCF)


def plot_det_curve(scores: np.ndarray, labels: np.ndarray, save_path: str = None):
    """DET (Detection Error Tradeoff) 곡선 그리기"""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    
    # DET 곡선은 log scale에서 그림
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr * 100, fnr * 100, 'b-', linewidth=2, label='DET curve')
    ax.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Random')
    
    # EER 지점 표시
    eer, _ = compute_eer(scores, labels)
    ax.plot(eer * 100, eer * 100, 'go', markersize=10, label=f'EER = {eer*100:.2f}%')
    
    ax.set_xlabel('False Positive Rate (%)')
    ax.set_ylabel('False Negative Rate (%)')
    ax.set_title('DET Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 DET curve saved to: {save_path}")
    
    plt.close()


def plot_score_distribution(
    scores: np.ndarray, 
    labels: np.ndarray, 
    save_path: str = None
):
    """점수 분포 히스토그램"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]
    
    ax.hist(bonafide_scores, bins=50, alpha=0.6, label='Bonafide', color='green', density=True)
    ax.hist(spoof_scores, bins=50, alpha=0.6, label='Spoof', color='red', density=True)
    
    ax.set_xlabel('Score (Spoof Probability)')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Score distribution saved to: {save_path}")
    
    plt.close()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict:
    """모델 평가"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Evaluating"):
            waveforms = waveforms.to(device)
            outputs = model(waveforms).squeeze()
            scores = torch.sigmoid(outputs).cpu().numpy()
            
            all_scores.extend(scores.tolist() if scores.ndim > 0 else [scores.item()])
            all_labels.extend(labels.numpy().tolist())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 메트릭 계산
    eer, eer_threshold = compute_eer(all_scores, all_labels)
    min_tdcf = compute_min_tdcf(all_scores, all_labels)
    auc = roc_auc_score(all_labels, all_scores)
    
    # 고정 threshold에서의 성능
    predictions = (all_scores > threshold).astype(int)
    accuracy = np.mean(predictions == all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # 추가 메트릭
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'min_tdcf': min_tdcf,
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'scores': all_scores.tolist(),
        'labels': all_labels.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Voice Phishing Detection Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                       help='Path to test data')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'rawnet2', 'aasist'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("📊 Voice Phishing Detection - Model Evaluation")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Data: {args.data_dir}")
    print(f"   Device: {args.device}")
    
    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    from src.data.synthetic_dataset import SyntheticDataset
    test_dataset = SyntheticDataset(args.data_dir, split='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"   Test samples: {len(test_dataset)}")
    
    # 모델 로드
    if args.model_type == 'simple':
        from scripts.quick_train import SimpleDetector
        model = SimpleDetector()
    elif args.model_type == 'rawnet2':
        from src.detection.detector import RawNet2
        model = RawNet2()
    elif args.model_type == 'aasist':
        from src.detection.detector import AASIST
        model = AASIST()
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)
    
    # 평가
    print("\n🔍 Running evaluation...")
    results = evaluate_model(model, test_loader, args.device, args.threshold)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📈 Evaluation Results")
    print("=" * 50)
    print(f"  EER: {results['eer']*100:.2f}%")
    print(f"  EER Threshold: {results['eer_threshold']:.4f}")
    print(f"  min-tDCF: {results['min_tdcf']:.4f}")
    print(f"  AUC-ROC: {results['auc']:.4f}")
    print(f"  Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Precision: {results['precision']*100:.2f}%")
    print(f"  Recall: {results['recall']*100:.2f}%")
    print(f"  F1 Score: {results['f1_score']*100:.2f}%")
    print("\n  Confusion Matrix:")
    print(f"    TN={results['true_negatives']}, FP={results['false_positives']}")
    print(f"    FN={results['false_negatives']}, TP={results['true_positives']}")
    
    # 결과 저장
    results_path = output_dir / 'evaluation_results.json'
    scores_labels = {
        'scores': results.pop('scores'),
        'labels': results.pop('labels')
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # 그래프 저장
    scores = np.array(scores_labels['scores'])
    labels = np.array(scores_labels['labels'])
    
    plot_det_curve(scores, labels, output_dir / 'det_curve.png')
    plot_score_distribution(scores, labels, output_dir / 'score_distribution.png')
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
