"""
Voice Phishing Detection - Data Download Script
ASVspoof, MLAAD, WaveFake 자동 다운로드
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
import zipfile
import tarfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def download_wavefake(data_dir: Path):
    """WaveFake 다운로드 (Zenodo)"""
    logger.info("📥 Downloading WaveFake dataset...")
    
    wavefake_dir = data_dir / "wavefake"
    wavefake_dir.mkdir(parents=True, exist_ok=True)
    
    # Zenodo Record ID: 5642694
    # 파일들: LJSpeech based fake samples
    urls = [
        ("https://zenodo.org/record/5642694/files/ljspeech_melgan.zip", "ljspeech_melgan.zip"),
        ("https://zenodo.org/record/5642694/files/ljspeech_full_band_melgan.zip", "ljspeech_full_band_melgan.zip"),
        ("https://zenodo.org/record/5642694/files/ljspeech_hifiGAN.zip", "ljspeech_hifiGAN.zip"),
        ("https://zenodo.org/record/5642694/files/ljspeech_parallel_wavegan.zip", "ljspeech_parallel_wavegan.zip"),
        ("https://zenodo.org/record/5642694/files/ljspeech_waveglow.zip", "ljspeech_waveglow.zip"),
    ]
    
    for url, filename in urls:
        zip_path = wavefake_dir / filename
        if not zip_path.exists():
            logger.info(f"  Downloading {filename}...")
            try:
                subprocess.run(
                    ["wget", "-q", "--show-progress", "-O", str(zip_path), url],
                    check=True
                )
            except subprocess.CalledProcessError:
                logger.warning(f"  Failed to download {filename}")
                continue
        
        # 압축 해제
        extract_dir = wavefake_dir / filename.replace(".zip", "")
        if not extract_dir.exists():
            logger.info(f"  Extracting {filename}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(wavefake_dir)
    
    # LJSpeech Real 다운로드 (for bonafide samples)
    real_dir = wavefake_dir / "ljspeech_real"
    if not real_dir.exists():
        logger.info("  Downloading LJSpeech real samples...")
        ljspeech_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        tar_path = wavefake_dir / "LJSpeech-1.1.tar.bz2"
        
        try:
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(tar_path), ljspeech_url],
                check=True
            )
            
            with tarfile.open(tar_path, 'r:bz2') as tf:
                tf.extractall(wavefake_dir)
            
            # wavs 폴더를 ljspeech_real로 이름 변경
            extracted = wavefake_dir / "LJSpeech-1.1" / "wavs"
            if extracted.exists():
                shutil.move(str(extracted), str(real_dir))
            
        except Exception as e:
            logger.warning(f"  Failed to download LJSpeech: {e}")
    
    logger.info(f"✅ WaveFake saved to: {wavefake_dir}")


def download_mlaad(data_dir: Path):
    """MLAAD 다운로드 (Hugging Face)"""
    logger.info("📥 Downloading MLAAD dataset...")
    
    mlaad_dir = data_dir / "mlaad"
    mlaad_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # MLAAD 다운로드 (일부만)
        logger.info("  Loading MLAAD from Hugging Face...")
        ds = load_dataset(
            "Classy-Cats/MLAAD-full",
            split="train[:10000]",  # 처음 10000개만
            trust_remote_code=True
        )
        
        # 로컬에 저장
        ds.save_to_disk(str(mlaad_dir))
        logger.info(f"✅ MLAAD saved to: {mlaad_dir}")
        
    except Exception as e:
        logger.warning(f"  Failed to download MLAAD: {e}")
        logger.info("  You can manually download from: https://huggingface.co/datasets/Classy-Cats/MLAAD-full")


def download_asvspoof_sample(data_dir: Path):
    """ASVspoof 2021 샘플 데이터 생성 (실제 데이터는 등록 필요)"""
    logger.info("📥 Creating ASVspoof sample data structure...")
    
    asvspoof_dir = data_dir / "asvspoof"
    asvspoof_dir.mkdir(parents=True, exist_ok=True)
    
    # 디렉토리 구조 생성
    subdirs = [
        "ASVspoof2021_LA_train/flac",
        "ASVspoof2021_LA_dev/flac",
        "ASVspoof2021_LA_eval/flac",
        "ASVspoof2021_LA_cm_protocols"
    ]
    
    for subdir in subdirs:
        (asvspoof_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 프로토콜 파일 예시 생성
    protocol_dir = asvspoof_dir / "ASVspoof2021_LA_cm_protocols"
    
    # Train protocol
    train_protocol = """LA_0001 LA_T_1000001 - - bonafide
LA_0001 LA_T_1000002 - A01 spoof
LA_0002 LA_T_1000003 - - bonafide
LA_0002 LA_T_1000004 - A02 spoof
LA_0003 LA_T_1000005 - - bonafide
"""
    with open(protocol_dir / "ASVspoof2021.LA.cm.train.trn.txt", 'w') as f:
        f.write(train_protocol)
    
    # Dev protocol
    with open(protocol_dir / "ASVspoof2021.LA.cm.dev.trl.txt", 'w') as f:
        f.write(train_protocol)  # 같은 형식 사용
    
    # Eval protocol
    with open(protocol_dir / "ASVspoof2021.LA.cm.eval.trl.txt", 'w') as f:
        f.write(train_protocol)
    
    logger.info(f"✅ ASVspoof structure created at: {asvspoof_dir}")
    logger.info("")
    logger.info("⚠️  ASVspoof 2021 실제 데이터는 등록 후 다운로드해야 합니다:")
    logger.info("   1. https://www.asvspoof.org/ 방문")
    logger.info("   2. EULA 동의 후 다운로드 링크 수신")
    logger.info(f"   3. 다운로드한 파일을 {asvspoof_dir}에 압축 해제")


def create_synthetic_samples(data_dir: Path, num_samples: int = 100):
    """합성 학습용 샘플 생성 (테스트용)"""
    logger.info("🔧 Creating synthetic samples for testing...")
    
    import numpy as np
    import soundfile as sf
    
    synthetic_dir = data_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    (synthetic_dir / "bonafide").mkdir(exist_ok=True)
    (synthetic_dir / "spoof").mkdir(exist_ok=True)
    
    sample_rate = 16000
    duration = 4.0
    samples_per_file = int(sample_rate * duration)
    
    for i in range(num_samples):
        # Bonafide: 자연스러운 음성 시뮬레이션 (노이즈 + 톤)
        t = np.linspace(0, duration, samples_per_file)
        freq = np.random.uniform(100, 300)  # 기본 주파수
        bonafide = 0.3 * np.sin(2 * np.pi * freq * t)
        bonafide += 0.1 * np.sin(2 * np.pi * freq * 2 * t)  # 배음
        bonafide += 0.05 * np.random.randn(samples_per_file)  # 노이즈
        bonafide = bonafide / np.max(np.abs(bonafide)) * 0.8
        
        sf.write(
            synthetic_dir / "bonafide" / f"bonafide_{i:04d}.wav",
            bonafide.astype(np.float32),
            sample_rate
        )
        
        # Spoof: TTS 특성 시뮬레이션 (더 clean, 반복적)
        freq = np.random.uniform(150, 250)
        spoof = 0.4 * np.sin(2 * np.pi * freq * t)
        spoof += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        spoof += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        spoof += 0.01 * np.random.randn(samples_per_file)  # 적은 노이즈
        spoof = spoof / np.max(np.abs(spoof)) * 0.9
        
        sf.write(
            synthetic_dir / "spoof" / f"spoof_{i:04d}.wav",
            spoof.astype(np.float32),
            sample_rate
        )
    
    # 프로토콜 파일 생성
    protocol_lines = []
    for i in range(num_samples):
        protocol_lines.append(f"S{i:04d} bonafide_{i:04d} - - bonafide")
        protocol_lines.append(f"S{i:04d} spoof_{i:04d} - A01 spoof")
    
    with open(synthetic_dir / "protocol.txt", 'w') as f:
        f.write('\n'.join(protocol_lines))
    
    logger.info(f"✅ Created {num_samples * 2} synthetic samples at: {synthetic_dir}")


def download_korean_samples(data_dir: Path):
    """한국어 음성 샘플 (AI Hub 안내)"""
    logger.info("📋 Korean voice datasets information...")
    
    korean_dir = data_dir / "korean"
    korean_dir.mkdir(parents=True, exist_ok=True)
    
    info = """# 한국어 음성 데이터셋 정보

## AI Hub 데이터셋 (회원가입 필요)
1. 자유대화 음성(일반남여)
   - URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=109
   
2. 명령어 음성(소음환경)
   - URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=571

3. 한국어 음성
   - URL: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123

## KsponSpeech (연구용)
- GitHub: https://github.com/sooftware/KsponSpeech
- 한국어 자발적 발화 데이터셋

## 다운로드 후 처리
1. AI Hub에서 다운로드
2. 이 디렉토리에 압축 해제
3. 오디오 파일을 16kHz로 리샘플링
"""
    
    with open(korean_dir / "README.md", 'w') as f:
        f.write(info)
    
    logger.info(f"✅ Korean dataset info saved to: {korean_dir}/README.md")


def main():
    parser = argparse.ArgumentParser(description='Download datasets for voice phishing detection')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to save datasets')
    parser.add_argument('--wavefake', action='store_true',
                       help='Download WaveFake dataset')
    parser.add_argument('--mlaad', action='store_true',
                       help='Download MLAAD dataset')
    parser.add_argument('--asvspoof', action='store_true',
                       help='Create ASVspoof directory structure')
    parser.add_argument('--synthetic', action='store_true',
                       help='Create synthetic test samples')
    parser.add_argument('--korean', action='store_true',
                       help='Show Korean dataset info')
    parser.add_argument('--all', action='store_true',
                       help='Download all available datasets')
    parser.add_argument('--num_synthetic', type=int, default=100,
                       help='Number of synthetic samples to create')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Voice Phishing Detection - Data Download")
    logger.info(f"📁 Data directory: {data_dir.absolute()}")
    
    if args.all or args.wavefake:
        download_wavefake(data_dir)
    
    if args.all or args.mlaad:
        download_mlaad(data_dir)
    
    if args.all or args.asvspoof:
        download_asvspoof_sample(data_dir)
    
    if args.all or args.synthetic:
        create_synthetic_samples(data_dir, args.num_synthetic)
    
    if args.all or args.korean:
        download_korean_samples(data_dir)
    
    if not any([args.all, args.wavefake, args.mlaad, args.asvspoof, args.synthetic, args.korean]):
        # 기본: 합성 샘플만 생성
        logger.info("No dataset specified. Creating synthetic samples for testing...")
        create_synthetic_samples(data_dir, args.num_synthetic)
    
    logger.info("\n✨ Done!")
    logger.info("Next steps:")
    logger.info("  1. python scripts/train_model.py --model rawnet2")
    logger.info("  2. python scripts/train_model.py --model aasist")


if __name__ == '__main__':
    main()
