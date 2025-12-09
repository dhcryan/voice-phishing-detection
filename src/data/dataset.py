"""
Voice Phishing Detection - Data Processing Pipeline
ASVspoof, MLAAD, WaveFake 데이터셋 처리
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """오디오 샘플 데이터 클래스"""
    file_path: str
    waveform: Optional[np.ndarray]
    label: int  # 0: bonafide, 1: spoof
    speaker_id: str
    attack_type: str
    dataset: str


class AudioPreprocessor:
    """오디오 전처리기"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration: float = 4.0,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(sample_rate * max_duration)
        self.normalize = normalize
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """오디오 파일 로드"""
        try:
            waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return waveform
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return np.zeros(self.max_samples)
    
    def pad_or_truncate(self, waveform: np.ndarray) -> np.ndarray:
        """길이 맞추기 (패딩 또는 자르기)"""
        if len(waveform) > self.max_samples:
            # 랜덤 시작점에서 자르기
            start = np.random.randint(0, len(waveform) - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
        elif len(waveform) < self.max_samples:
            # 제로 패딩
            padding = self.max_samples - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
        return waveform
    
    def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """오디오 정규화"""
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def process(self, file_path: str) -> np.ndarray:
        """전체 전처리 파이프라인"""
        waveform = self.load_audio(file_path)
        waveform = self.pad_or_truncate(waveform)
        if self.normalize:
            waveform = self.normalize_audio(waveform)
        return waveform.astype(np.float32)


class ASVspoofDataset(Dataset):
    """ASVspoof 2021 LA 데이터셋"""
    
    def __init__(
        self,
        audio_dir: str,
        protocol_file: str,
        preprocessor: AudioPreprocessor,
        split: str = "train"
    ):
        self.audio_dir = Path(audio_dir)
        self.preprocessor = preprocessor
        self.split = split
        
        # 프로토콜 파일 파싱
        self.samples = self._load_protocol(protocol_file)
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_protocol(self, protocol_file: str) -> List[Dict]:
        """ASVspoof 프로토콜 파일 파싱"""
        samples = []
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    samples.append({
                        'speaker_id': parts[0],
                        'file_name': parts[1],
                        'attack_type': parts[3],
                        'label': 0 if parts[4] == 'bonafide' else 1
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # 오디오 파일 경로 (flac 또는 wav)
        for ext in ['.flac', '.wav']:
            audio_path = self.audio_dir / f"{sample['file_name']}{ext}"
            if audio_path.exists():
                break
        
        # 전처리
        waveform = self.preprocessor.process(str(audio_path))
        
        return torch.from_numpy(waveform), sample['label']


class MLAADDataset(Dataset):
    """MLAAD 데이터셋 (Hugging Face)"""
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        max_samples: int = None
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.samples = self._load_samples(max_samples)
        logger.info(f"Loaded {len(self.samples)} MLAAD samples")
    
    def _load_samples(self, max_samples: int = None) -> List[Dict]:
        """MLAAD 샘플 로드"""
        samples = []
        
        # Arrow 파일에서 로드 (Hugging Face datasets 저장 형식)
        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(self.data_dir))
            
            for i, item in enumerate(ds):
                if max_samples and i >= max_samples:
                    break
                    
                samples.append({
                    'audio': item['audio'],
                    'label': item.get('label', 1),  # MLAAD는 모두 fake
                    'language': item.get('language', 'unknown'),
                    'model': item.get('model', 'unknown')
                })
        except Exception as e:
            logger.warning(f"Failed to load MLAAD: {e}")
            # 폴백: 디렉토리에서 직접 로드
            for audio_file in self.data_dir.glob("**/*.wav"):
                samples.append({
                    'file_path': str(audio_file),
                    'label': 1,  # fake
                    'language': 'unknown',
                    'model': 'unknown'
                })
                if max_samples and len(samples) >= max_samples:
                    break
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        if 'audio' in sample and isinstance(sample['audio'], dict):
            # Hugging Face 오디오 형식
            waveform = np.array(sample['audio']['array'], dtype=np.float32)
            waveform = self.preprocessor.pad_or_truncate(waveform)
            if self.preprocessor.normalize:
                waveform = self.preprocessor.normalize_audio(waveform)
        else:
            # 파일 경로
            waveform = self.preprocessor.process(sample['file_path'])
        
        return torch.from_numpy(waveform), sample['label']


class WaveFakeDataset(Dataset):
    """WaveFake 데이터셋"""
    
    def __init__(
        self,
        data_dir: str,
        preprocessor: AudioPreprocessor,
        include_real: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.samples = self._load_samples(include_real)
        logger.info(f"Loaded {len(self.samples)} WaveFake samples")
    
    def _load_samples(self, include_real: bool) -> List[Dict]:
        """WaveFake 샘플 로드"""
        samples = []
        
        # Fake 샘플들 (각 TTS/VC 폴더)
        fake_dirs = [
            'ljspeech_melgan', 'ljspeech_melgan_large', 
            'ljspeech_full_band_melgan', 'ljspeech_multi_band_melgan',
            'ljspeech_hifiGAN', 'ljspeech_waveglow',
            'ljspeech_parallel_wavegan', 'ljspeech_mb_melgan'
        ]
        
        for fake_dir in fake_dirs:
            dir_path = self.data_dir / fake_dir
            if dir_path.exists():
                for audio_file in dir_path.glob("*.wav"):
                    samples.append({
                        'file_path': str(audio_file),
                        'label': 1,  # spoof
                        'attack_type': fake_dir
                    })
        
        # Real 샘플들
        if include_real:
            real_dir = self.data_dir / 'ljspeech_real'
            if real_dir.exists():
                for audio_file in real_dir.glob("*.wav"):
                    samples.append({
                        'file_path': str(audio_file),
                        'label': 0,  # bonafide
                        'attack_type': 'real'
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        waveform = self.preprocessor.process(sample['file_path'])
        return torch.from_numpy(waveform), sample['label']


class CombinedDataset(Dataset):
    """여러 데이터셋 결합"""
    
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
    
    def __len__(self) -> int:
        return sum(self.lengths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 어느 데이터셋에서 가져올지 결정
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")


def create_dataloaders(
    asvspoof_dir: str = None,
    mlaad_dir: str = None,
    wavefake_dir: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_duration: float = 4.0
) -> Dict[str, DataLoader]:
    """데이터로더 생성"""
    
    preprocessor = AudioPreprocessor(
        sample_rate=sample_rate,
        max_duration=max_duration
    )
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # ASVspoof 2021
    if asvspoof_dir and Path(asvspoof_dir).exists():
        asvspoof_path = Path(asvspoof_dir)
        
        # Train
        train_audio = asvspoof_path / "ASVspoof2021_LA_train" / "flac"
        train_protocol = asvspoof_path / "ASVspoof2021_LA_cm_protocols" / "ASVspoof2021.LA.cm.train.trn.txt"
        if train_audio.exists() and train_protocol.exists():
            train_datasets.append(ASVspoofDataset(train_audio, train_protocol, preprocessor, "train"))
        
        # Dev
        dev_audio = asvspoof_path / "ASVspoof2021_LA_dev" / "flac"
        dev_protocol = asvspoof_path / "ASVspoof2021_LA_cm_protocols" / "ASVspoof2021.LA.cm.dev.trl.txt"
        if dev_audio.exists() and dev_protocol.exists():
            val_datasets.append(ASVspoofDataset(dev_audio, dev_protocol, preprocessor, "dev"))
        
        # Eval
        eval_audio = asvspoof_path / "ASVspoof2021_LA_eval" / "flac"
        eval_protocol = asvspoof_path / "ASVspoof2021_LA_cm_protocols" / "ASVspoof2021.LA.cm.eval.trl.txt"
        if eval_audio.exists() and eval_protocol.exists():
            test_datasets.append(ASVspoofDataset(eval_audio, eval_protocol, preprocessor, "eval"))
    
    # MLAAD
    if mlaad_dir and Path(mlaad_dir).exists():
        mlaad_ds = MLAADDataset(mlaad_dir, preprocessor, max_samples=10000)
        # 8:1:1 분할
        n = len(mlaad_ds)
        train_n = int(n * 0.8)
        val_n = int(n * 0.1)
        indices = np.random.permutation(n)
        
        train_indices = indices[:train_n]
        val_indices = indices[train_n:train_n+val_n]
        test_indices = indices[train_n+val_n:]
        
        train_datasets.append(torch.utils.data.Subset(mlaad_ds, train_indices))
        val_datasets.append(torch.utils.data.Subset(mlaad_ds, val_indices))
        test_datasets.append(torch.utils.data.Subset(mlaad_ds, test_indices))
    
    # WaveFake
    if wavefake_dir and Path(wavefake_dir).exists():
        wavefake_ds = WaveFakeDataset(wavefake_dir, preprocessor)
        n = len(wavefake_ds)
        train_n = int(n * 0.8)
        val_n = int(n * 0.1)
        indices = np.random.permutation(n)
        
        train_indices = indices[:train_n]
        val_indices = indices[train_n:train_n+val_n]
        test_indices = indices[train_n+val_n:]
        
        train_datasets.append(torch.utils.data.Subset(wavefake_ds, train_indices))
        val_datasets.append(torch.utils.data.Subset(wavefake_ds, val_indices))
        test_datasets.append(torch.utils.data.Subset(wavefake_ds, test_indices))
    
    # 데이터로더 생성
    dataloaders = {}
    
    if train_datasets:
        train_combined = CombinedDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        dataloaders['train'] = DataLoader(
            train_combined, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
    
    if val_datasets:
        val_combined = CombinedDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        dataloaders['val'] = DataLoader(
            val_combined, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    if test_datasets:
        test_combined = CombinedDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
        dataloaders['test'] = DataLoader(
            test_combined, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    return dataloaders


if __name__ == "__main__":
    # 테스트
    print("📦 데이터 파이프라인 테스트")
    
    preprocessor = AudioPreprocessor()
    
    # 더미 오디오 생성 테스트
    dummy_audio = np.random.randn(16000).astype(np.float32)
    processed = preprocessor.pad_or_truncate(dummy_audio)
    print(f"✅ 전처리 테스트: {len(dummy_audio)} -> {len(processed)} samples")
