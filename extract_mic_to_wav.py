import pyedflib
import numpy as np
import scipy.io.wavfile as wavfile
import os

def extract_mic_to_wav_auto(edf_path, output_folder):
    # 提取原文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(edf_path))[0]
    output_path = os.path.join(output_folder, base_name + ".wav")

    # 打开 EDF 文件
    f = pyedflib.EdfReader(edf_path)
    labels = f.getSignalLabels()

    if "Mic" not in labels:
        raise ValueError("Mic channel not found in EDF file.")

    mic_index = labels.index("Mic")
    sample_rate = int(f.getSampleFrequency(mic_index))
    signal = f.readSignal(mic_index)
    f.close()

    # 归一化并转换为 16-bit PCM 格式
    signal = signal / np.max(np.abs(signal))  # 归一化到 [-1, 1]
    signal_int16 = (signal * 32767).astype(np.int16)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存为 WAV
    wavfile.write(output_path, sample_rate, signal_int16)
    print(f"Saved to {output_path}")