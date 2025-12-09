"""
Voice Phishing Detection - Model Training Script
학습, 검증, 평가 전체 파이프라인
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders, AudioPreprocessor
from src.detection.detector import RawNet2, AASIST, ECAPATDNNDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """조기 종료"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class FocalLoss(nn.Module):
    """클래스 불균형을 위한 Focal Loss"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Equal Error Rate 계산"""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    try:
        # 유효한 범위 확인
        if len(fpr) < 2:
            return 0.5, 0.5
        eer_threshold = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0, 1)
        eer = interp1d(fpr, fnr)(eer_threshold)
    except (ValueError, Exception):
        eer = 0.5
        eer_threshold = 0.5
    
    return float(eer), float(eer_threshold)


def compute_tdcf(scores: np.ndarray, labels: np.ndarray, 
                 Pspoof: float = 0.05, Cmiss: float = 1, Cfa: float = 10) -> float:
    """tandem Detection Cost Function (t-DCF) 계산"""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # min t-DCF 계산
    Ptar = 1 - Pspoof
    tDCF = Cmiss * fnr * Ptar + Cfa * fpr * Pspoof
    min_tDCF = np.min(tDCF)
    
    return float(min_tDCF)


class Trainer:
    """모델 학습기"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = True,
        use_amp: bool = True,
        checkpoint_dir: str = 'checkpoints',
        experiment_name: str = 'experiment'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 스케줄러 (Cosine Annealing with Warm Restarts)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # 손실 함수
        self.criterion = FocalLoss() if use_focal_loss else nn.BCEWithLogitsLoss()
        
        # Mixed Precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # 체크포인트
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # 학습 기록
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_eer': [], 'val_tdcf': []
        }
        self.best_eer = float('inf')
    
    def train_epoch(self) -> Tuple[float, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (waveforms, labels) in enumerate(pbar):
            waveforms = waveforms.to(self.device)
            labels = labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(waveforms).squeeze()
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(waveforms).squeeze()
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Dict:
        """검증/평가"""
        self.model.eval()
        loader = loader or self.val_loader
        
        total_loss = 0.0
        all_scores = []
        all_labels = []
        
        for waveforms, labels in tqdm(loader, desc="Validating"):
            waveforms = waveforms.to(self.device)
            labels_tensor = labels.float().to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(waveforms).squeeze()
                    loss = self.criterion(outputs, labels_tensor)
            else:
                outputs = self.model(waveforms).squeeze()
                loss = self.criterion(outputs, labels_tensor)
            
            total_loss += loss.item()
            scores = torch.sigmoid(outputs).cpu().numpy()
            all_scores.extend(scores.tolist() if scores.ndim > 0 else [scores.item()])
            all_labels.extend(labels.numpy().tolist())
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 메트릭 계산
        avg_loss = total_loss / len(loader)
        accuracy = np.mean((all_scores > 0.5) == all_labels)
        eer, eer_threshold = compute_eer(all_scores, all_labels)
        min_tdcf = compute_tdcf(all_scores, all_labels)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'min_tdcf': min_tdcf
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        # 최신 체크포인트
        path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        torch.save(checkpoint, path)
        
        # 최고 성능 체크포인트
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Best model saved: EER={metrics['eer']:.4f}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(
        self,
        num_epochs: int = 100,
        patience: int = 15,
        save_every: int = 5
    ) -> Dict:
        """전체 학습 루프"""
        early_stopping = EarlyStopping(patience=patience)
        
        logger.info(f"🚀 Training started: {self.experiment_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Epochs: {num_epochs}, Patience: {patience}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_metrics = self.validate()
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 기록
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_eer'].append(val_metrics['eer'])
            self.history['val_tdcf'].append(val_metrics['min_tdcf'])
            
            # 로깅
            logger.info(f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%")
            logger.info(f"Val EER: {val_metrics['eer']*100:.2f}%, min-tDCF: {val_metrics['min_tdcf']:.4f}")
            
            # 체크포인트 저장
            is_best = val_metrics['eer'] < self.best_eer
            if is_best:
                self.best_eer = val_metrics['eer']
            
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 조기 종료
            if early_stopping(val_metrics['loss']):
                logger.info(f"⏹️ Early stopping at epoch {epoch}")
                break
        
        # 최종 테스트
        if self.test_loader:
            logger.info("\n🧪 Final evaluation on test set...")
            # 최고 모델 로드
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
            if best_path.exists():
                self.load_checkpoint(str(best_path))
            
            test_metrics = self.validate(self.test_loader)
            logger.info(f"Test EER: {test_metrics['eer']*100:.2f}%")
            logger.info(f"Test min-tDCF: {test_metrics['min_tdcf']:.4f}")
            
            return test_metrics
        
        return val_metrics


def create_model(model_name: str, device: str = 'cuda') -> nn.Module:
    """모델 생성"""
    if model_name == 'rawnet2':
        model = RawNet2(
            sinc_out_channels=128,
            first_conv_out=128,
            gru_hidden=1024,
            gru_layers=3,
            fc_hidden=1024
        )
    elif model_name == 'aasist':
        model = AASIST(
            sinc_out_channels=70,
            encoder_channels=[128, 256, 512],
            graph_hidden=128,
            num_heads=4
        )
    elif model_name == 'ecapa':
        model = ECAPATDNNDetector(model_name="speechbrain/spkrec-ecapa-voxceleb")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Voice Phishing Detection Training')
    
    # 데이터 경로
    parser.add_argument('--asvspoof_dir', type=str, default='data/asvspoof',
                       help='ASVspoof 2021 dataset directory')
    parser.add_argument('--mlaad_dir', type=str, default='data/mlaad',
                       help='MLAAD dataset directory')
    parser.add_argument('--wavefake_dir', type=str, default='data/wavefake',
                       help='WaveFake dataset directory')
    
    # 모델 설정
    parser.add_argument('--model', type=str, default='rawnet2',
                       choices=['rawnet2', 'aasist', 'ecapa'],
                       help='Model architecture')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    
    # 오디오 설정
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--max_duration', type=float, default=4.0)
    
    # 기타
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 실험 이름
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"🎯 Experiment: {args.experiment_name}")
    logger.info(f"📊 Model: {args.model}")
    logger.info(f"💻 Device: {args.device}")
    
    # 데이터로더 생성
    logger.info("📦 Loading datasets...")
    dataloaders = create_dataloaders(
        asvspoof_dir=args.asvspoof_dir,
        mlaad_dir=args.mlaad_dir,
        wavefake_dir=args.wavefake_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )
    
    if not dataloaders:
        logger.error("❌ No datasets found! Please download datasets first.")
        logger.info("Run: ./scripts/download_data.sh")
        return
    
    logger.info(f"✅ Train: {len(dataloaders.get('train', []))} batches")
    logger.info(f"✅ Val: {len(dataloaders.get('val', []))} batches")
    logger.info(f"✅ Test: {len(dataloaders.get('test', []))} batches")
    
    # 모델 생성
    logger.info(f"🔧 Creating model: {args.model}")
    model = create_model(args.model, args.device)
    
    # 파라미터 수
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📐 Trainable parameters: {num_params:,}")
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        train_loader=dataloaders.get('train'),
        val_loader=dataloaders.get('val'),
        test_loader=dataloaders.get('test'),
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name
    )
    
    # 학습 시작
    final_metrics = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # 결과 저장
    results_path = Path(args.checkpoint_dir) / f"{args.experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'final_metrics': {k: float(v) for k, v in final_metrics.items()},
            'history': trainer.history
        }, f, indent=2)
    
    logger.info(f"\n✨ Training completed!")
    logger.info(f"📁 Results saved to: {results_path}")


if __name__ == '__main__':
    main()
