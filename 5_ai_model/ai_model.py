import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# 경로 설정
dataset_dir = "./mel_images"
img_height, img_width = 224, 224
batch_size = 32

# 이미지 전처리
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# 학습 데이터
train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 검증 데이터
val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 클래스 수
num_classes = train_data.num_classes

# 모델 구성
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# 모델 저장 (.keras 파일 생성, .h5도 가능)
model.save("noise_classifier_cnn.keras")
print("✅ 모델이 'noise_classifier_cnn.keras'로 저장되었습니다.")