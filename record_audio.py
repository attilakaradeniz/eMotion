import os
import time
import winsound
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import logging
from tqdm import tqdm

def record_audio(duration=10):
    try:
        # Klasörü oluştur
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.abspath(os.path.join("recordings", f"recording_{timestamp}.wav"))

        # Kayıt parametreleri
        samplerate = 44100
        channels = 1

        # Geri sayım
        print("\nRecording will start in:")
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        winsound.Beep(1000, 500)
        print("\nRecording... Speak now!")

        # Kayda başla
        recording = sd.rec(int(duration * samplerate),
                           samplerate=samplerate,
                           channels=channels,
                           dtype=np.int16)

        # tqdm ilerleme çubuğu
        with tqdm(total=duration, desc="Recording", unit="sec",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} seconds") as pbar:
            for _ in range(duration):
                time.sleep(1)
                pbar.update(1)

        sd.wait()
        winsound.Beep(1000, 500)
        print("\nRecording finished!")

        # WAV olarak kaydet
        sf.write(filename, recording, samplerate)

        logging.info(f"Recording saved: {filename}")
        print(f"✅ Saved to: {filename}")
        return filename

    except Exception as e:
        logging.error(f"Error during recording: {e}")
        print("\n❌ Error during recording:")
        print(e)
        raise
