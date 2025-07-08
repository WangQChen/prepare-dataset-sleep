import numpy as np
import librosa
import pandas as pd
import os
import tqdm
import scipy

def extract_features(audio, sr=16000):
    hop_length = int(0.015 * sr)
    frame_length = int(0.06 * sr)
    features = {}

    def enhanced_autocorr(x, interval_len):
        if len(x) < interval_len:
            x = np.pad(x, (0, interval_len - len(x)))
        S = librosa.stft(x[:interval_len], n_fft=1024, hop_length=hop_length)
        mag = np.abs(S)

        def acf(vec):
            return np.correlate(vec, vec, mode='full')[len(vec)//2:]

        ac_list = [acf(mag[i]) for i in range(mag.shape[0])]
        max_peaks = [np.max(a[1:]) if len(a) > 1 else 0 for a in ac_list]
        top_indices = np.argsort(max_peaks)[-int(0.5 * len(max_peaks)):]
        R = np.mean([ac_list[i] for i in top_indices], axis=0)

        peak_idx, _ = scipy.signal.find_peaks(R)
        if len(peak_idx) < 1:
            return np.nan, np.nan, np.nan
        cp = peak_idx[0]
        ci = R[cp]
        cc = np.std(np.diff(peak_idx)) if len(peak_idx) > 1 else 0
        return cp, ci, cc
        # 呼吸周期,呼吸周期强度,呼吸周期一致性

    # 呼吸节律特征（12s与24s窗口）
    interval_12 = min(len(audio), int(12 * sr))
    interval_24 = min(len(audio), int(24 * sr))

    features['CP_12'], features['CI_12'], features['CC_12'] = enhanced_autocorr(audio, interval_12)
    features['CP_24'], features['CI_24'], features['CC_24'] = enhanced_autocorr(audio, interval_24)

    # 鼾声特征,使用 librosa.feature.rms 计算音频信号的“短时能量” 来衡量鼾声的强度
    # 最大鼾声概率,每小时鼾声次数估计
    rms_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    features['Max_SLS'] = np.max(rms_energy)
    threshold = np.median(rms_energy) + 1.5 * np.std(rms_energy)
    snore_count = np.sum(rms_energy > threshold)
    features['Snore_Index'] = snore_count * (60 / (len(rms_energy) * hop_length / sr))

    return features

def process_long_audio_to_epochs(wav_path, epoch_sec=30, sr=16000, output_csv='features.csv'):
    audio, sr = librosa.load(wav_path, sr=sr)
    epoch_len = int(epoch_sec * sr)
    total_epochs = len(audio) // epoch_len

    feature_rows = []
    for i in tqdm.tqdm(range(total_epochs), desc="Processing epochs"):
        start_sample = i * epoch_len
        end_sample = start_sample + epoch_len
        epoch_audio = audio[start_sample:end_sample]
        feats = extract_features(epoch_audio, sr=sr)
        feats['start_time_sec'] = i * epoch_sec
        feature_rows.append(feats)

    df = pd.DataFrame(feature_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ 提取完成，共 {total_epochs} 个 epoch，结果保存到 {output_csv}")
