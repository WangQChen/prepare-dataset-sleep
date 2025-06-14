import pyedflib

def check_edf_channel_auto(edf_path):
    f = pyedflib.EdfReader(edf_path)
    n_channels = f.signals_in_file
    labels = f.getSignalLabels()

    print(f"共有 {n_channels} 个通道:")
    for i, label in enumerate(labels):
        print(f"{i+1}. {label}")

    f.close()