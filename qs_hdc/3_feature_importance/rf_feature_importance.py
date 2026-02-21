import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchvision.datasets import MNIST
from torchhd.datasets.isolet import ISOLET
from torchhd.datasets.ucihar import UCIHAR
from torchhd.datasets import EMGHandGestures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


def compute_importances(X, y, feature_names=None, test_size=0.2, random_state=42, n_estimators=400, max_depth=None, max_features="sqrt", n_jobs=-1):
    print(f"[RF] Starting importance computation (n_samples={X.shape[0]}, n_features={X.shape[1]}, n_estimators={n_estimators})", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"[RF] Training RandomForest on {X_train.shape[0]} samples...", flush=True)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=n_jobs, random_state=random_state)
    rf.fit(X_train, y_train)
    impurity_imp = rf.feature_importances_
    print(f"[RF] Computing permutation importances on {X_test.shape[0]} held-out samples...", flush=True)
    perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=1)
    perm_mean = perm.importances_mean
    perm_std = perm.importances_std
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame({"feature": feature_names, "impurity_importance": impurity_imp, "perm_importance_mean": perm_mean, "perm_importance_std": perm_std})
    df["rank_perm"] = df["perm_importance_mean"].rank(ascending=False, method="min")
    df["rank_impurity"] = df["impurity_importance"].rank(ascending=False, method="min")
    df["rank_avg"] = (df["rank_perm"] + df["rank_impurity"]) / 2.0
    df_sorted = df.sort_values(by=["rank_perm", "rank_avg"], ascending=[True, True]).reset_index(drop=True)
    return rf, df_sorted

def load_mnist(data_root):
    transform = torchvision.transforms.ToTensor()
    train_ds = MNIST(data_root, train=True, transform=transform, download=True)
    X = train_ds.data.numpy().astype(np.float32).reshape(train_ds.data.shape[0], -1) / 255.0
    y = train_ds.targets.numpy().astype(np.int64)
    feat_names = [f"px_{i}" for i in range(X.shape[1])]
    return X, y, feat_names

def load_isolet(root):
    train_ds = ISOLET(root=root, train=True, download=True)
    X = train_ds.data.numpy().astype(np.float32)
    y = train_ds.targets.numpy().astype(np.int64)
    feat_names = [f"iso_{i}" for i in range(X.shape[1])]
    return X, y, feat_names

def load_ucihar(root):
    train_ds = UCIHAR(root=root, train=True, download=True)
    X = train_ds.data.numpy().astype(np.float32)
    y = train_ds.targets.numpy().astype(np.int64)
    feat_names = [f"har_{i}" for i in range(X.shape[1])]
    return X, y, feat_names

def load_emg(root):
    def transform(x):
        return x.flatten()
    train_ds = EMGHandGestures(root, subjects=[0, 1, 2, 3], download=True, transform=transform)
    X_list = []
    y_list = []
    for samples, labels in train_ds:
        X_list.append(np.asarray(samples, dtype=np.float32))
        y_list.append(int(labels))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    feat_names = [f"emg_{i}" for i in range(X.shape[1])]
    return X, y, feat_names

def save_results(df, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

def save_mnist_heatmap(df, feat_names, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    vals = df.set_index("feature")["perm_importance_mean"].reindex(feat_names).values
    heat = vals.reshape(28, 28)
    plt.figure(figsize=(5, 5))
    sns.heatmap(heat, cmap="magma")
    plt.title("MNIST Permutation Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def run_isolet(data_root, results_dir, n_estimators):
    print("[Dataset] ISOLET: loading data...", flush=True)
    X, y, feat_names = load_isolet(data_root)
    print(f"[Dataset] ISOLET: data loaded (n_samples={X.shape[0]}, n_features={X.shape[1]})", flush=True)
    _, df = compute_importances(X, y, feature_names=feat_names, n_estimators=n_estimators)
    save_results(df, os.path.join(results_dir, "isolet_feature_importance.csv"))

def run_mnist(data_root, results_dir, n_estimators):
    print("[Dataset] MNIST: loading data...", flush=True)
    X, y, feat_names = load_mnist(data_root)
    print(f"[Dataset] MNIST: data loaded (n_samples={X.shape[0]}, n_features={X.shape[1]})", flush=True)
    _, df = compute_importances(X, y, feature_names=feat_names, n_estimators=n_estimators)
    save_results(df, os.path.join(results_dir, "mnist_feature_importance.csv"))
    save_mnist_heatmap(df, feat_names, os.path.join(results_dir, "mnist_perm_heatmap.png"))

def run_har(data_root, results_dir, n_estimators):
    print("[Dataset] UCIHAR: loading data...", flush=True)
    X, y, feat_names = load_ucihar(data_root)
    print(f"[Dataset] UCIHAR: data loaded (n_samples={X.shape[0]}, n_features={X.shape[1]})", flush=True)
    _, df = compute_importances(X, y, feature_names=feat_names, n_estimators=n_estimators)
    save_results(df, os.path.join(results_dir, "har_feature_importance.csv"))

def run_emg(data_root, results_dir, n_estimators):
    print("[Dataset] EMG: loading data...", flush=True)
    X, y, feat_names = load_emg(data_root)
    print(f"[Dataset] EMG: data loaded (n_samples={X.shape[0]}, n_features={X.shape[1]})", flush=True)
    _, df = compute_importances(X, y, feature_names=feat_names, n_estimators=n_estimators)
    save_results(df, os.path.join(results_dir, "emg_feature_importance.csv"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["isolet", "mnist", "har", "emg", "all"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./rf_feature_importance_results")
    parser.add_argument("--n_estimators", type=int, default=400)
    args = parser.parse_args()
    if args.dataset in ["isolet", "all"]:
        run_isolet(args.data_root, args.results_dir, args.n_estimators)
    if args.dataset in ["mnist", "all"]:
        run_mnist(args.data_root, args.results_dir, args.n_estimators)
    if args.dataset in ["har", "all"]:
        run_har(args.data_root, args.results_dir, args.n_estimators)
    if args.dataset in ["emg", "all"]:
        run_emg(args.data_root, args.results_dir, args.n_estimators)

if __name__ == "__main__":
    main()
