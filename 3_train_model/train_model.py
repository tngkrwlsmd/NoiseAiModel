import os
import csv
from scipy.io.wavfile import read
from scipy.fft import fft
import numpy as np
import shutil

# 설정
input_dir = "./sample/wav"
output_dir = "./dataset"
csv_file = "results.csv"

os.makedirs(output_dir, exist_ok=True)

def get_dominant_freq(file_path):
    rate, data = read(file_path)
    if data.ndim > 1:
        data = data[:, 0]
    N = len(data)
    T = 1.0 / rate
    yf = fft(data)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    y_abs = 2.0 / N * np.abs(yf[0:N // 2])
    return xf[np.argmax(y_abs)]

def label_freq(freq):
    if freq < 250:
        return "low_freq"
    elif freq > 2000:
        return "high_freq"
    else:
        return "mid_freq"

# CSV 파일 초기화
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Original File", "Dominant Frequency (Hz)", "Label", "New Path"])

    # 입력 디렉토리의 모든 .wav 파일 처리
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            freq = get_dominant_freq(file_path)
            label = label_freq(freq)

            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            count = len(os.listdir(label_dir))
            new_filename = f"{label}_{count + 1:03d}.wav"
            new_path = os.path.join(label_dir, new_filename)

            shutil.copy(file_path, new_path)

            writer.writerow([filename, f"{freq:.2f}", label, new_path])
            print(f"✅ {filename} → {label}/{new_filename} ({freq:.2f} Hz)")