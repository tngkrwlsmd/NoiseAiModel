import sounddevice as sd
from scipy.io.wavfile import write

# 마이크로 소음을 채집하는 코드입니다. 샘플 소음을 생성하기 위해서라면 ./sample/sample_wav.py 파일을 실행해주세요.

def record_audio(filename='./sample/output.wav', duration=5, samplerate=44100):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    write(filename, samplerate, audio)
    print(f"✅ Recording saved as {filename}")

record_audio()