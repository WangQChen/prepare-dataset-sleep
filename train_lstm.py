import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from torch.utils.data import ConcatDataset

# ---------- 常量设置 ----------
LABEL_MAP_RAW_TO_CLEAN = {
    "Wake": "Wake",
    "REM": "REM",
    "NonREM1": "NonREM",
    "NonREM2": "NonREM",
    "NonREM3": "NonREM"
}
LABEL_ENCODED = {
    "Wake": 0,
    "NonREM": 1,
    "REM": 2
}

# ---------- Dataset ----------
class SleepSequenceDataset(Dataset):
    def __init__(self, df, window_size=5):
        self.window = window_size
        self.half = window_size // 2
        self.features = df[["CP_12","CI_12","CC_12","CP_24","CI_24","CC_24","Max_SLS","Snore_Index"]].values.astype("float32")
        self.labels = df["stage"].map(LABEL_ENCODED).values.astype("int64")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        start = max(0, idx - self.half)
        end = min(len(self.features), idx + self.half + 1)
        seq = self.features[start:end]
        if len(seq) < self.window:
            pad = np.zeros((self.window - len(seq), seq.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq]) if start == 0 else np.vstack([seq, pad])
            # print(seq.shape)
            # seq.shape = (5, 8)，5 是时间步长（时序长度），8 是每个 epoch 的特征维度
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ---------- Model ----------
class LSTMClassifier(nn.Module):

    def __init__(self, input_dim=8, hidden_dim=64, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ---------- Data Loading ----------
def load_subject_data(subject_id, feat_path="data/feature", label_path="data/csv"):
    df_feat = pd.read_csv(f"{feat_path}/{subject_id}.csv")
    df_label = pd.read_csv(f"{label_path}/{subject_id}/{subject_id}_user.csv")

    # 清洗标签：NonREM1/2/3 → NonREM
    df_label["stage"] = df_label["stage"].map(LABEL_MAP_RAW_TO_CLEAN)

    # 删除无效标签（如 map 后为 NaN 的行）
    df_label = df_label[df_label["stage"].isin(LABEL_ENCODED.keys())]

    df = pd.merge(df_feat, df_label, left_on="start_time_sec", right_on="start")
    df["subject"] = subject_id
    return df

def load_all_data():
    subject_ids = [os.path.basename(f).replace(".csv", "") for f in glob.glob("data/feature/*.csv")]
    data = pd.concat([load_subject_data(sid) for sid in subject_ids], ignore_index=True)
    groups = data["subject"].astype("category").cat.codes
    return data, groups, subject_ids

# ---------- Train & Eval ----------
def train_one_fold(train_loader, val_loader, device, fold_name="model.pt"):
    model = LSTMClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), fold_name)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1

# ---------- Cross-validation ----------
def run_cross_subject():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all, groups, subject_ids = load_all_data()
    logo = LeaveOneGroupOut()
    acc_list, f1_list = [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(df_all, df_all["stage"], groups)):
        train_df = df_all.iloc[train_idx]
        test_df = df_all.iloc[test_idx]
        train_set = SleepSequenceDataset(train_df)
        test_set = SleepSequenceDataset(test_df)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=64)

        acc, f1 = train_one_fold(train_loader, test_loader, device, fold_name=f"model_fold{fold+1}.pt")
        print(f"[Subject {subject_ids[fold]}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        acc_list.append(acc)
        f1_list.append(f1)

    print(f"\n✅ Average Accuracy: {np.mean(acc_list):.4f}")
    print(f"✅ Average F1-score: {np.mean(f1_list):.4f}")


def load_model_for_inference(model_path, device):
    model = LSTMClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_final_dataset():
    subject_ids = [os.path.basename(f).replace(".csv", "") for f in glob.glob("data/feature/*.csv")]
    datasets = []

    for sid in subject_ids:
        df = load_subject_data(sid)
        dataset = SleepSequenceDataset(df)
        datasets.append(dataset)

    return ConcatDataset(datasets)

def train_final_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_dataset = build_final_dataset()
    final_loader = DataLoader(final_dataset, batch_size=64, shuffle=True)

    model = LSTMClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for X, y in final_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "final_model.pt")
    print("✅ Final model saved to final_model.pt")

def evaluate_final_model_on_each_subject():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = LSTMClassifier().to(device)
    model.load_state_dict(torch.load("final_model.pt", map_location=device))
    model.eval()

    subject_ids = [os.path.basename(f).replace(".csv", "") for f in glob.glob("data/feature/*.csv")]
    acc_list, f1_list = [], []

    for sid in subject_ids:
        df = load_subject_data(sid)
        dataset = SleepSequenceDataset(df)
        loader = DataLoader(dataset, batch_size=64)

        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = np.mean(np.array(y_true) == np.array(y_pred))
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"[Subject {sid}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        acc_list.append(acc)
        f1_list.append(f1)

    print(f"\n✅ Average Accuracy: {np.mean(acc_list):.4f}")
    print(f"✅ Average F1-score: {np.mean(f1_list):.4f}")


# ---------- Run ----------
if __name__ == "__main__":
    run_cross_subject()