import numpy as np
import os
from scipy.io.wavfile import write

# 저장할 디렉토리
output_dir = './sample/wav'
os.makedirs(output_dir, exist_ok=True)

# 오디오 기본 설정
sample_rate = 44100  # 44.1kHz
duration = 1.0       # 1초

# 생성할 샘플 수
num_samples_per_class = 333

# 주파수 구간 정의
freq_ranges = {
    'low': (50, 250),
    'mid': (250, 2000),
    'high': (2000, 10000)
}

# 각 구간에서 균등하게 .wav 파일 생성
for label, (f_min, f_max) in freq_ranges.items():
    for i in range(num_samples_per_class):
        freq = np.random.uniform(f_min, f_max)  # 주파수 무작위 선택
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * freq * t)

        # 파일명: 예) low_0.wav
        filename = f"{label}_{i}.wav"
        filepath = os.path.join(output_dir, filename)

        # int16 형식으로 변환 후 저장
        scaled_waveform = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
        write(filepath, sample_rate, scaled_waveform)

print("모든 .wav 파일 생성 완료 ✅")