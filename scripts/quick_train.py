#!/usr/bin/env python
"""
Quick Train Script - 합성 데이터로 빠른 학습 테스트
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.data.synthetic_dataset import create_synthetic_dataloaders


class SimpleDetector(nn.Module):
    """간단한 CNN 기반 탐지기 (테스트용)"""
    
    def __init__(self, input_samples: int = 64000):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=251, stride=1, padding=125),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, samples) -> (batch, 1, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)
        x = self.fc(x)  # (batch, 1)
        
        return x


def train():
    print("🚀 Quick Training on Synthetic Data")
    
    # 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    # 데이터 로드
    data_dir = project_root / "data" / "synthetic"
    if not data_dir.exists():
        print(f"❌ No synthetic data found at {data_dir}")
        print("Run: python scripts/download_data.py --synthetic")
        return
    
    print(f"📦 Loading data from {data_dir}")
    dataloaders = create_synthetic_dataloaders(
        data_dir=str(data_dir),
        batch_size=16,
        num_workers=2,
        sample_rate=16000,
        max_duration=4.0
    )
    
    print(f"   Train: {len(dataloaders['train'])} batches")
    print(f"   Val: {len(dataloaders['val'])} batches")
    print(f"   Test: {len(dataloaders['test'])} batches")
    
    # 모델
    model = SimpleDetector().to(device)
    print(f"📐 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 학습 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for waveforms, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch}"):
            waveforms = waveforms.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for waveforms, labels in dataloaders['val']:
                waveforms = waveforms.to(device)
                labels = labels.float().to(device)
                
                outputs = model(waveforms).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch}: Train Loss={train_loss/len(dataloaders['train']):.4f}, "
              f"Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # 모델 저장
            checkpoint_dir = project_root / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "simple_detector_best.pt")
    
    # Test
    print("\n🧪 Testing...")
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for waveforms, labels in dataloaders['test']:
            waveforms = waveforms.to(device)
            labels = labels.float().to(device)
            
            outputs = model(waveforms).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    print(f"✅ Test Accuracy: {test_acc*100:.1f}%")
    print(f"💾 Best model saved to: checkpoints/simple_detector_best.pt")


if __name__ == '__main__':
    train()
