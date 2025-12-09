#!/bin/bash
# ==========================================
# Voice Phishing Detection - Data Download Script
# ASVspoof, MLAAD, WaveFake 데이터셋 다운로드
# ==========================================

set -e

DATA_DIR="/home/dhc99/voice-phishing-detection/data/audio"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=========================================="
echo "🎤 음성 데이터셋 다운로드 안내"
echo "=========================================="

# ==========================================
# 1. ASVspoof 2021 (공식)
# ==========================================
echo ""
echo "📦 1. ASVspoof 2021 Dataset"
echo "   - 공식 사이트: https://www.asvspoof.org/index2021.html"
echo "   - 다운로드 방법:"
echo "     1) 위 사이트 방문"
echo "     2) 'Download' 섹션에서 등록"
echo "     3) LA (Logical Access) 트랙 다운로드"
echo ""
echo "   - 직접 다운로드 (등록 후 받은 링크 사용):"
echo "     wget --user=YOUR_ID --password=YOUR_PW <다운로드_링크>"
echo ""

# ASVspoof 2021 평가 프로토콜 (GitHub에서 공개)
echo "   📥 ASVspoof 2021 프로토콜/메타데이터 다운로드 중..."
if [ ! -d "asvspoof2021_protocols" ]; then
    git clone https://github.com/asvspoof-challenge/2021.git asvspoof2021_protocols 2>/dev/null || echo "   ⚠️ Git clone 실패 - 수동 다운로드 필요"
fi

# ==========================================
# 2. MLAAD (Hugging Face)
# ==========================================
echo ""
echo "📦 2. MLAAD Dataset (Multi-Language Audio Anti-Spoofing)"
echo "   - Hugging Face: https://huggingface.co/datasets/Habs/MLAAD"
echo "   - 23개 언어, 52개 TTS 모델"
echo ""
echo "   📥 Hugging Face에서 다운로드:"
echo "   pip install datasets"
echo "   python -c \"from datasets import load_dataset; ds = load_dataset('Habs/MLAAD', split='train'); ds.save_to_disk('$DATA_DIR/mlaad')\""
echo ""

# Python으로 MLAAD 다운로드 시도
read -p "   MLAAD 데이터셋을 다운로드할까요? (y/n): " download_mlaad
if [ "$download_mlaad" = "y" ]; then
    echo "   📥 MLAAD 다운로드 중... (시간이 걸릴 수 있습니다)"
    python3 << 'EOF'
try:
    from datasets import load_dataset
    print("   Loading MLAAD from Hugging Face...")
    # 작은 샘플만 먼저 다운로드
    ds = load_dataset("Habs/MLAAD", split="train", streaming=True)
    sample = list(ds.take(100))
    print(f"   ✅ 샘플 {len(sample)}개 로드 성공!")
    print("   전체 데이터셋은 용량이 크므로 필요시 전체 다운로드하세요.")
except ImportError:
    print("   ⚠️ 'datasets' 패키지가 필요합니다: pip install datasets")
except Exception as e:
    print(f"   ⚠️ 다운로드 실패: {e}")
EOF
fi

# ==========================================
# 3. WaveFake (Zenodo)
# ==========================================
echo ""
echo "📦 3. WaveFake Dataset"
echo "   - 논문: https://arxiv.org/abs/2111.02813"
echo "   - 다운로드: https://zenodo.org/record/5642694"
echo ""
echo "   📥 다운로드 명령어:"
echo "   wget https://zenodo.org/record/5642694/files/wavefake.zip"
echo "   unzip wavefake.zip -d $DATA_DIR/wavefake"
echo ""

read -p "   WaveFake 데이터셋을 다운로드할까요? (y/n): " download_wavefake
if [ "$download_wavefake" = "y" ]; then
    echo "   📥 WaveFake 다운로드 중..."
    wget -c https://zenodo.org/record/5642694/files/generated_audio.zip -O wavefake.zip 2>/dev/null || echo "   ⚠️ 다운로드 실패"
    if [ -f "wavefake.zip" ]; then
        unzip -q wavefake.zip -d wavefake 2>/dev/null || echo "   ⚠️ 압축 해제 실패"
        echo "   ✅ WaveFake 다운로드 완료!"
    fi
fi

# ==========================================
# 4. 한국어 음성 데이터 (AI Hub)
# ==========================================
echo ""
echo "📦 4. 한국어 음성 데이터 (선택사항)"
echo "   - AI Hub: https://aihub.or.kr"
echo "   - '한국어 음성' 검색 후 다운로드"
echo "   - 회원가입 및 승인 필요"
echo ""

# ==========================================
# 5. 샘플 테스트 오디오 생성
# ==========================================
echo ""
echo "📦 5. 테스트용 샘플 오디오 생성"
echo ""

read -p "   테스트용 샘플 오디오를 생성할까요? (y/n): " create_sample
if [ "$create_sample" = "y" ]; then
    python3 << 'EOF'
import numpy as np
import os

try:
    import scipy.io.wavfile as wav
    
    # 간단한 테스트 오디오 생성 (1초 사인파)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 음
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    output_path = "test_sample.wav"
    wav.write(output_path, sample_rate, audio)
    print(f"   ✅ 테스트 오디오 생성: {output_path}")
    
except ImportError:
    print("   ⚠️ scipy 필요: pip install scipy")
except Exception as e:
    print(f"   ⚠️ 생성 실패: {e}")
EOF
fi

# ==========================================
# 완료
# ==========================================
echo ""
echo "=========================================="
echo "✅ 데이터 다운로드 안내 완료!"
echo "=========================================="
echo ""
echo "📁 데이터 디렉토리: $DATA_DIR"
echo ""
echo "📋 요약:"
echo "   1. ASVspoof 2021: 공식 사이트 등록 후 다운로드"
echo "   2. MLAAD: Hugging Face datasets 라이브러리 사용"
echo "   3. WaveFake: Zenodo에서 직접 다운로드"
echo "   4. 한국어 데이터: AI Hub에서 신청"
echo ""
ls -la "$DATA_DIR" 2>/dev/null || echo "(디렉토리 비어있음)"
