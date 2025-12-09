"""
Voice Phishing Detection - Detection Module
Implements various anti-spoofing models: AASIST, RawNet2, ECAPA-TDNN, Wav2Vec2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result from anti-spoofing model"""
    is_fake: bool
    fake_probability: float
    confidence: float
    model_name: str
    processing_time_ms: float
    raw_scores: Dict[str, float]
    embeddings: Optional[np.ndarray] = None


class AudioPreprocessor:
    """Audio preprocessing utilities"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: int = 60):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = sample_rate * max_duration
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and normalize audio file"""
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Truncate if too long
        if len(waveform) > self.max_samples:
            waveform = waveform[:self.max_samples]
        
        # Normalize
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        
        return waveform, sr
    
    def extract_mel_spectrogram(
        self, 
        waveform: np.ndarray,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160
    ) -> np.ndarray:
        """Extract mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec
    
    def extract_mfcc(
        self, 
        waveform: np.ndarray,
        n_mfcc: int = 40
    ) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=n_mfcc
        )
        return mfcc
    
    def extract_acoustic_features(self, waveform: np.ndarray) -> Dict[str, float]:
        """Extract acoustic anomaly features (jitter, shimmer, etc.)"""
        features = {}
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=self.sample_rate)
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=self.sample_rate)
        features["rolloff_mean"] = float(np.mean(rolloff))
        
        # RMS energy
        rms = librosa.feature.rms(y=waveform)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))
        
        # Spectral flatness (indicates synthetic artifacts)
        flatness = librosa.feature.spectral_flatness(y=waveform)
        features["flatness_mean"] = float(np.mean(flatness))
        
        return features


class BaseDetector(ABC):
    """Base class for anti-spoofing detectors"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.preprocessor = AudioPreprocessor()
        self.model_name = "base"

    @abstractmethod
    def load_model(self):
        """Load model weights"""
        pass
    
    @abstractmethod
    def predict(self, audio_path: str) -> DetectionResult:
        """Run prediction"""
        pass
    
    def to_tensor(self, waveform: np.ndarray) -> torch.Tensor:
        """Convert waveform to tensor"""
        return torch.from_numpy(waveform).unsqueeze(0).to(self.device)


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


class SimpleTestDetector(BaseDetector):
    """Simple CNN Detector for testing pipeline"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.model_name = "simple_cnn"
        
    def load_model(self):
        self.model = SimpleDetector().to(self.device)
        
        if self.model_path and Path(self.model_path).exists():
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded SimpleDetector weights from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
        else:
            logger.warning(f"Model path {self.model_path} not found, using random weights")
        
        self.model.eval()
        
    def predict(self, audio_path: str) -> DetectionResult:
        import time
        start_time = time.time()
        
        # Load audio
        waveform, sr = self.preprocessor.load_audio(audio_path)
        acoustic_features = self.preprocessor.extract_acoustic_features(waveform)
        
        # Prepare input (fixed length for simple model)
        target_len = 64000  # 4 seconds
        if len(waveform) > target_len:
            waveform = waveform[:target_len]
        else:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
            
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(waveform_tensor).squeeze()
            fake_prob = torch.sigmoid(logits).item()
            
        is_fake = fake_prob > 0.5
        confidence = max(fake_prob, 1 - fake_prob)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_fake=is_fake,
            fake_probability=fake_prob,
            confidence=confidence,
            model_name=self.model_name,
            processing_time_ms=processing_time,
            raw_scores={
                "fake_score": fake_prob,
                **acoustic_features
            },
            embeddings=None
        )
        
    @abstractmethod
    def load_model(self):
        """Load pretrained model"""
        pass
    
    @abstractmethod
    def predict(self, audio_path: str) -> DetectionResult:
        """Run prediction on audio file"""
        pass
    
    def to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor"""
        tensor = torch.from_numpy(array).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)


class SincConv(nn.Module):
    """Sinc-based convolution layer for RawNet2"""
    
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: float = 50,
        min_band_hz: float = 50
    ):
        super().__init__()
        
        # Ensure odd kernel size for symmetry
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
            
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize filterbank
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        mel = np.linspace(
            self._hz_to_mel(low_hz),
            self._hz_to_mel(high_hz),
            out_channels + 1
        )
        hz = self._mel_to_hz(mel)
        
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        n_ = (kernel_size - 1) / 2.0
        self.n_ = 2 * np.pi * torch.arange(-n_, 0).view(1, -1) / sample_rate
        
    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, waveform):
        self.n_ = self.n_.to(waveform.device)
        
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2
        )
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / 
            (self.n_ / 2)
        ) * 2
        
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(waveform, filters, stride=1, padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    """Residual block for RawNet2"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.mp = nn.MaxPool1d(3)
        self.relu = nn.LeakyReLU(0.3)
        
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out = out + identity
        out = self.mp(out)
        
        return out


class RawNet2(nn.Module):
    """
    RawNet2 Model for Audio Anti-Spoofing
    Reference: https://arxiv.org/abs/2011.01108
    """
    
    def __init__(
        self,
        sinc_out_channels: int = 128,
        sinc_kernel_size: int = 1024,
        res_channels: List[int] = [128, 256, 256, 512, 512],
        gru_hidden: int = 1024,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Sinc convolution layer
        self.sinc_conv = SincConv(
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size
        )
        self.bn_sinc = nn.BatchNorm1d(sinc_out_channels)
        self.relu = nn.LeakyReLU(0.3)
        self.mp = nn.MaxPool1d(3)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_ch = sinc_out_channels
        for out_ch in res_channels:
            self.res_blocks.append(ResBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=res_channels[-1],
            hidden_size=gru_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )
        
        # Classifier
        self.fc = nn.Linear(gru_hidden, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, samples) or (batch, samples)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Sinc convolution
        x = self.sinc_conv(x)
        x = self.bn_sinc(x)
        x = self.relu(x)
        x = self.mp(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # GRU
        x = x.transpose(1, 2)  # (batch, time, channels)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Last hidden state
        
        # Classification
        out = self.fc(x)
        
        return out
    
    def extract_embedding(self, x):
        """Extract embedding before classification layer"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.sinc_conv(x)
        x = self.bn_sinc(x)
        x = self.relu(x)
        x = self.mp(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        embedding = x[:, -1, :]
        
        return embedding


class RawNet2Detector(BaseDetector):
    """RawNet2 based anti-spoofing detector"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.model_name = "RawNet2"
        
    def load_model(self):
        """Load RawNet2 model"""
        self.model = RawNet2().to(self.device)
        
        if self.model_path and Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded RawNet2 weights from {self.model_path}")
        else:
            logger.warning("Using randomly initialized RawNet2 model")
        
        self.model.eval()
        
    def predict(self, audio_path: str) -> DetectionResult:
        """Run prediction on audio file"""
        import time
        start_time = time.time()
        
        # Load and preprocess audio
        waveform, sr = self.preprocessor.load_audio(audio_path)
        waveform_tensor = self.to_tensor(waveform)
        
        # Get acoustic features
        acoustic_features = self.preprocessor.extract_acoustic_features(waveform)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(waveform_tensor)
            probs = F.softmax(logits, dim=1)
            embedding = self.model.extract_embedding(waveform_tensor)
        
        # Get prediction
        fake_prob = probs[0, 1].item()  # Class 1 = fake
        is_fake = fake_prob > 0.5
        confidence = max(fake_prob, 1 - fake_prob)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_fake=is_fake,
            fake_probability=fake_prob,
            confidence=confidence,
            model_name=self.model_name,
            processing_time_ms=processing_time,
            raw_scores={
                "real_score": probs[0, 0].item(),
                "fake_score": probs[0, 1].item(),
                **acoustic_features
            },
            embeddings=embedding.cpu().numpy()
        )


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for AASIST"""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj=None):
        # x: (batch, nodes, features)
        batch_size, n_nodes, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # (batch, nodes, out_features)
        h = h.view(batch_size, n_nodes, self.n_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)  # (batch, heads, nodes, head_dim)
        
        # Compute attention scores
        # Self-attention for all pairs
        h_repeat = h.unsqueeze(3).expand(-1, -1, -1, n_nodes, -1)
        h_repeat_interleave = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)
        concat = torch.cat([h_repeat, h_repeat_interleave], dim=-1)
        
        e = self.leaky_relu(torch.einsum('bhnmf,hf->bhnm', concat, self.a))
        
        # Apply mask if adjacency provided
        if adj is not None:
            e = e.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))
        
        attention = F.softmax(e, dim=-1)
        
        # Aggregate
        out = torch.einsum('bhnm,bhmd->bhnd', attention, h)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, n_nodes, -1)
        
        return out


class AASIST(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
    Reference: https://arxiv.org/abs/2110.01200
    """
    
    def __init__(
        self,
        sinc_out_channels: int = 70,
        sinc_kernel_size: int = 128,
        encoder_hidden: int = 64,
        graph_hidden: int = 64,
        n_heads: int = 4,
        num_classes: int = 2
    ):
        super().__init__()
        
        # Sinc convolution for filterbank
        self.sinc_conv = SincConv(
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size
        )
        self.bn_sinc = nn.BatchNorm1d(sinc_out_channels)
        
        # Spectral encoder
        self.spectral_encoder = nn.Sequential(
            nn.Conv2d(1, encoder_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_hidden),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(encoder_hidden, encoder_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Graph attention layers
        self.gat1 = GraphAttentionLayer(encoder_hidden, graph_hidden, n_heads)
        self.gat2 = GraphAttentionLayer(graph_hidden, graph_hidden, n_heads)
        
        # Readout and classifier
        self.readout = nn.Sequential(
            nn.Linear(graph_hidden * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Sinc filterbank
        x = self.sinc_conv(x)
        x = self.bn_sinc(x)
        x = F.relu(x)
        
        # Reshape for 2D convolution
        batch_size = x.shape[0]
        x = x.unsqueeze(1)  # Add channel dim
        
        # Spectral encoding
        x = self.spectral_encoder(x)
        
        # Reshape for graph attention
        x = x.view(batch_size, -1, 64).permute(0, 2, 1)  # (batch, 64, hidden)
        
        # Graph attention
        x = self.gat1(x)
        x = F.relu(x)
        x = self.gat2(x)
        
        # Readout
        x = x.view(batch_size, -1)
        out = self.readout(x)
        
        return out


class AASISTDetector(BaseDetector):
    """AASIST based anti-spoofing detector"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.model_name = "AASIST"
        
    def load_model(self):
        """Load AASIST model"""
        self.model = AASIST().to(self.device)
        
        if self.model_path and Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded AASIST weights from {self.model_path}")
        else:
            logger.warning("Using randomly initialized AASIST model")
        
        self.model.eval()
        
    def predict(self, audio_path: str) -> DetectionResult:
        """Run prediction on audio file"""
        import time
        start_time = time.time()
        
        # Load and preprocess audio
        waveform, sr = self.preprocessor.load_audio(audio_path)
        waveform_tensor = self.to_tensor(waveform)
        
        # Get acoustic features
        acoustic_features = self.preprocessor.extract_acoustic_features(waveform)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(waveform_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Get prediction
        fake_prob = probs[0, 1].item()
        is_fake = fake_prob > 0.5
        confidence = max(fake_prob, 1 - fake_prob)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_fake=is_fake,
            fake_probability=fake_prob,
            confidence=confidence,
            model_name=self.model_name,
            processing_time_ms=processing_time,
            raw_scores={
                "real_score": probs[0, 0].item(),
                "fake_score": probs[0, 1].item(),
                **acoustic_features
            }
        )


class ECAPATDNNDetector(BaseDetector):
    """ECAPA-TDNN based detector using SpeechBrain embeddings"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        super().__init__(model_path, device)
        self.model_name = "ECAPA-TDNN"
        self.embedding_model = None
        self.classifier = None
        
    def load_model(self):
        """Load ECAPA-TDNN from SpeechBrain"""
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/ecapa",
                run_opts={"device": str(self.device)}
            )
            
            # Simple classifier on top of embeddings
            self.classifier = nn.Sequential(
                nn.Linear(192, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            ).to(self.device)
            
            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.classifier.load_state_dict(checkpoint)
                logger.info(f"Loaded ECAPA classifier weights from {self.model_path}")
            else:
                logger.warning("Using randomly initialized ECAPA classifier")
            
            self.classifier.eval()
            
        except ImportError:
            logger.error("SpeechBrain not installed. Run: pip install speechbrain")
            raise
            
    def predict(self, audio_path: str) -> DetectionResult:
        """Run prediction on audio file"""
        import time
        start_time = time.time()
        
        # Load audio
        waveform, sr = self.preprocessor.load_audio(audio_path)
        acoustic_features = self.preprocessor.extract_acoustic_features(waveform)
        
        # Get embedding
        waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.embedding_model.encode_batch(waveform_tensor)
            embedding = embedding.squeeze(1)
            
            logits = self.classifier(embedding)
            probs = F.softmax(logits, dim=1)
        
        fake_prob = probs[0, 1].item()
        is_fake = fake_prob > 0.5
        confidence = max(fake_prob, 1 - fake_prob)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            is_fake=is_fake,
            fake_probability=fake_prob,
            confidence=confidence,
            model_name=self.model_name,
            processing_time_ms=processing_time,
            raw_scores={
                "real_score": probs[0, 0].item(),
                "fake_score": probs[0, 1].item(),
                **acoustic_features
            },
            embeddings=embedding.cpu().numpy()
        )


def get_detector(model_type: str, model_path: Optional[str] = None, device: str = "cuda") -> BaseDetector:
    """Factory function to get detector by type"""
    detectors = {
        "aasist": AASISTDetector,
        "rawnet2": RawNet2Detector,
        "ecapa": ECAPATDNNDetector,
        "simple": SimpleTestDetector,
    }
    
    if model_type not in detectors:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(detectors.keys())}")
    
    detector = detectors[model_type](model_path, device)
    detector.load_model()
    
    return detector
