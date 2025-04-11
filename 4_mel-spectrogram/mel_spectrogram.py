import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def convert_to_mel_images(source_dir="dataset", output_dir="mel_images"):
    os.makedirs(output_dir, exist_ok=True)
    for label in os.listdir(source_dir):
        label_path = os.path.join(source_dir, label)
        if not os.path.isdir(label_path):
            continue

        out_label_path = os.path.join(output_dir, label)
        os.makedirs(out_label_path, exist_ok=True)

        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                file_path = os.path.join(label_path, file)
                y, sr = librosa.load(file_path, sr=None)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                plt.figure(figsize=(2.24, 2.24))  # 정사각형 이미지 (224x224용)
                librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, cmap='magma')
                plt.axis('off')

                image_filename = os.path.splitext(file)[0] + ".png"
                out_path = os.path.join(out_label_path, image_filename)
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"✅ Saved: {out_path}")

convert_to_mel_images()