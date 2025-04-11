import numpy as np
from scipy.io.wavfile import read
from scipy.fft import fft
import os
import csv

# 🔍 주파수 분석 함수
def analyze_frequency(file_path):
    rate, data = read(file_path)
    if data.ndim > 1:
        data = data[:, 0]  # 스테레오면 첫 채널만 사용

    N = len(data)
    T = 1.0 / rate
    yf = fft(data)
    xf = np.fft.fftfreq(N, T)[:N//2]
    y_abs = 2.0 / N * np.abs(yf[0:N//2])

    dominant_freq = xf[np.argmax(y_abs)]

    if dominant_freq < 250:
        category = "Low Frequency"
    elif dominant_freq > 2000:
        category = "High Frequency"
    else:
        category = "Mid Frequency"

    return os.path.basename(file_path), dominant_freq, category

# 📁 분석 실행 & CSV 저장
sample_dir = "./sample/wav"
output_csv = "frequency_analysis.csv"

results = []

# 모든 .wav 파일 분석
for fname in sorted(os.listdir(sample_dir)):
    if fname.endswith(".wav"):
        file_path = os.path.join(sample_dir, fname)
        filename, freq, category = analyze_frequency(file_path)
        print(f"📈 {filename} - {freq:.2f} Hz ({category})")
        results.append([filename, f"{freq:.2f}", category])

# CSV 파일로 저장
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Dominant Frequency (Hz)", "Category"])
    writer.writerows(results)

print(f"\n✅ 분석 결과가 '{output_csv}'로 저장되었습니다.")