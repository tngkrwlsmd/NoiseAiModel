import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


# 마이크를 통해 소리를 채집하는 코드, 테스트 중에는 랜덤 주파수를 생성하도록 한다
#def record_audio(duration=DURATION, samplerate=SAMPLERATE):
#    print("🎤 Listening...")
#    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
#    sd.wait()
#    return np.squeeze(audio)

# 모델 및 파라미터
model = load_model("noise_classifier_cnn.keras")
IMG_SIZE = (224, 224)
SAMPLERATE = 22050
DURATION = 2  # 초
class_names = ['high_freq', 'low_freq', 'mid_freq'] # 실제 클래스 순서에 맞게!

def generate_sine_wave(freq, duration=DURATION, samplerate=SAMPLERATE):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)

def audio_to_mel_image(audio):
    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLERATE, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)  # 이건 거의 224x224 사이즈
    librosa.display.specshow(S_dB, sr=SAMPLERATE, cmap='magma')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.resize((224, 224), resample=Image.Resampling.BILINEAR)
    buf.close()
    return np.array(img) / 255.0

def predict_from_sine(freq):
    audio = generate_sine_wave(freq)
    mel_img = audio_to_mel_image(audio)
    input_data = np.expand_dims(mel_img, axis=0)
    pred = model.predict(input_data)

    idx = np.argmax(pred)
    return class_names[idx], pred[0][idx]

# 저주파 (50~250Hz) 10개 + 중간파 (250~1000Hz) 10개 + 고주파 (2000~10000Hz) 10개 → 총 30개 랜덤 추출
random.seed(42)
low_freqs = random.sample(range(50, 251), 10)
mid_freps = random.sample(range(250, 2001), 10)
high_freqs = random.sample(range(2000, 10001), 10)

test_freqs = sorted(low_freqs + mid_freps + high_freqs)
results = []

for freq in test_freqs:
    label, conf = predict_from_sine(freq)
    results.append((freq, label, conf))
    print(f"📈 {freq}Hz → {label} ({conf*100:.2f}%)")

# 결과 시각화
freqs = [r[0] for r in results]
labels = [r[1] for r in results]
confidences = [r[2] for r in results]

# 클래스별 색상 지정
color_map = {
    'low_freq': 'blue',
    'mid_freq': 'green',
    'high_freq': 'red'
}
colors = [color_map.get(lbl, 'gray') for lbl in labels]

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(freqs, confidences, color=colors, width=80)
plt.xlabel("주파수 (Hz)")
plt.ylabel("신뢰도")
plt.title("주파수별 분류 결과 (빨강: high, 초록: mid, 파랑: low)")
plt.xticks(freqs, rotation=45)
plt.grid(True)

# 범례 수동 추가
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color='red', label='high_freq'),
    mpatches.Patch(color='green', label='mid_freq'),
    mpatches.Patch(color='blue', label='low_freq')
]
plt.legend(handles=legend_handles)

plt.tight_layout()
plt.show()