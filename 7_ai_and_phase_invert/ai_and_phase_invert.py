import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 설정
SAMPLE_RATE = 22050
DURATION = 2
IMG_SIZE = (224, 224)
class_names = ['high_freq', 'low_freq', 'mid_freq']  # 분류 클래스

# 모델 불러오기
model = load_model("noise_classifier_cnn.keras")

# 소음 채집 코드
#def record_audio(duration=DURATION):
#    print("🎤 소리 녹음 중...")
#    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
#    sd.wait()
#    return np.squeeze(audio)

# 랜덤 주파수 오디오 생성 (녹음 대신)
def generate_random_audio():
    freq_ranges = {
        'low_freq': (50, 250),
        'mid_freq': (250, 2000),
        'high_freq': (2000, 10000)
    }
    label = random.choice(list(freq_ranges.keys()))  # 대역 중 하나 선택
    low, high = freq_ranges[label]
    freq = random.uniform(low, high)  # 선택된 대역 내에서 주파수 생성

    print(f"🎵 테스트용 랜덤 주파수 생성: {label} ({freq:.2f}Hz)")

    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.float32), label

# Mel-spectrogram → 이미지 변환
def audio_to_mel_image(audio):
    S = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_dB, sr=SAMPLE_RATE, cmap='magma')
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize(IMG_SIZE)
    buf.close()

    return np.array(img) / 255.0

# 분류
def predict_class(audio):
    mel_img = audio_to_mel_image(audio)
    input_data = np.expand_dims(mel_img, axis=0)
    prediction = model.predict(input_data, verbose=0)
    idx = np.argmax(prediction)
    label = class_names[idx]
    confidence = prediction[0][idx]
    print(f"🧠 예측: {label} ({confidence * 100:.2f}%)")
    return label

# 위상 반전 출력
def phase_invert_and_play(audio):
    print("🔄 위상 반전 후 출력 중...")
    inverted = -1 * audio
    sd.play(inverted, samplerate=SAMPLE_RATE)
    sd.wait()

# 혼동 행렬 테스트 실행
true_labels = []
pred_labels = []
NUM_SAMPLES = 50

print(f"\n📊 {NUM_SAMPLES}개의 랜덤 오디오에 대해 예측 및 혼동 행렬 계산...\n")

for i in range(NUM_SAMPLES):
    print(f"[{i+1}/{NUM_SAMPLES}]")
    audio, true_label = generate_random_audio()
    predicted_label = predict_class(audio)

    true_labels.append(true_label)
    pred_labels.append(predicted_label)

    # 위상 반전 출력 (원할 경우 주석 해제)
    # if predicted_label in class_names:
    #     phase_invert_and_play(audio)

    print("-" * 30)
    time.sleep(0.2)

print(classification_report(true_labels, pred_labels, target_names=class_names))

# 혼동 행렬 출력(모델이 정확하게 분류했는지 확인)
cm = confusion_matrix(true_labels, pred_labels, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("혼동 행렬 (Confusion Matrix)")
plt.show()