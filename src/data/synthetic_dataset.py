"""
Voice Phishing Detection - Synthetic Dataset for Training
합성 데이터를 사용한 학습용 데이터셋
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
from typing import Tuple, Dict, List


class SyntheticDataset(Dataset):
    """합성 테스트 데이터셋"""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 4.0,
        split: str = "train"
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.split = split
        
        # 샘플 로드
        self.samples = self._load_samples()
        
        # train/val/test 분할 (8:1:1)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        n = len(indices)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:train_end]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[train_end:val_end]]
        else:  # test
            self.samples = [self.samples[i] for i in indices[val_end:]]
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        
        # Bonafide 샘플
        bonafide_dir = self.data_dir / "bonafide"
        if bonafide_dir.exists():
            for f in bonafide_dir.glob("*.wav"):
                samples.append({"path": str(f), "label": 0})
        
        # Spoof 샘플
        spoof_dir = self.data_dir / "spoof"
        if spoof_dir.exists():
            for f in spoof_dir.glob("*.wav"):
                samples.append({"path": str(f), "label": 1})
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # 오디오 로드
        waveform, _ = librosa.load(sample["path"], sr=self.sample_rate, mono=True)
        
        # 패딩/자르기
        if len(waveform) > self.max_samples:
            start = np.random.randint(0, len(waveform) - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        elif len(waveform) < self.max_samples:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
        
        # 정규화
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        return torch.from_numpy(waveform.astype(np.float32)), sample["label"]


class LJSpeechDataset(Dataset):
    """LJSpeech 기반 데이터셋 (Real samples)"""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 4.0,
        max_samples: int = None,
        label: int = 0  # 0=bonafide
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_audio_samples = int(sample_rate * max_duration)
        self.label = label
        
        # 오디오 파일 찾기
        self.audio_files = list(self.data_dir.glob("**/*.wav"))
        
        if max_samples:
            self.audio_files = self.audio_files[:max_samples]
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.audio_files[idx]
        
        waveform, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        
        # 패딩/자르기
        if len(waveform) > self.max_audio_samples:
            start = np.random.randint(0, len(waveform) - self.max_audio_samples)
            waveform = waveform[start:start + self.max_audio_samples]
        elif len(waveform) < self.max_audio_samples:
            waveform = np.pad(waveform, (0, self.max_audio_samples - len(waveform)))
        
        # 정규화
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        return torch.from_numpy(waveform.astype(np.float32)), self.label


def create_synthetic_dataloaders(
    data_dir: str = "data/synthetic",
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_duration: float = 4.0
) -> Dict[str, DataLoader]:
    """합성 데이터용 데이터로더 생성"""
    
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = SyntheticDataset(
            data_dir=data_dir,
            sample_rate=sample_rate,
            max_duration=max_duration,
            split=split
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


def create_ljspeech_dataloaders(
    real_dir: str,
    fake_dirs: List[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_duration: float = 4.0,
    max_samples_per_class: int = 5000
) -> Dict[str, DataLoader]:
    """LJSpeech 기반 데이터로더 생성"""
    from torch.utils.data import ConcatDataset, random_split
    
    datasets = []
    
    # Real samples
    if Path(real_dir).exists():
        real_ds = LJSpeechDataset(
            real_dir, sample_rate, max_duration, 
            max_samples=max_samples_per_class, label=0
        )
        datasets.append(real_ds)
    
    # Fake samples from various TTS
    if fake_dirs:
        for fake_dir in fake_dirs:
            if Path(fake_dir).exists():
                fake_ds = LJSpeechDataset(
                    fake_dir, sample_rate, max_duration,
                    max_samples=max_samples_per_class // len(fake_dirs), label=1
                )
                datasets.append(fake_ds)
    
    if not datasets:
        return {}
    
    # 합치기
    full_dataset = ConcatDataset(datasets)
    
    # 분할
    n = len(full_dataset)
    train_n = int(n * 0.8)
    val_n = int(n * 0.1)
    test_n = n - train_n - val_n
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )
    
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    }
