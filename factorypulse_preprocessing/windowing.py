"""
把一段長訊號切成固定長度、可重疊的時間窗。

重要：切窗只是把「同一檔案」切成很多小樣本，train/val/test 一定要按「檔案」切分，
不能先切窗再隨機分樣本，不然同一段錄音的窗會同時出現在 train 和 test，
準確率會虛高（data leakage）。這件事在 build_dataset.py 裡處理，這裡只負責切。
"""

import numpy as np


def sliding_window(signal: np.ndarray, fs: float, window_sec: float, overlap: float) -> np.ndarray:
    """
    signal: shape (N,) 或 (N, C)
    回傳: shape (n_windows, window_len) 或 (n_windows, window_len, C)
    """
    window_len = int(round(window_sec * fs))
    step = int(round(window_len * (1 - overlap)))
    if step <= 0:
        raise ValueError("overlap 太大，step <= 0")

    n = signal.shape[0]
    starts = list(range(0, n - window_len + 1, step))
    if not starts:
        raise ValueError(
            f"訊號長度 {n} 點，小於一個窗需要的 {window_len} 點 "
            f"(window_sec={window_sec}, fs={fs})"
        )

    windows = np.stack([signal[s : s + window_len] for s in starts], axis=0)
    return windows


def n_windows_for(signal_len: int, fs: float, window_sec: float, overlap: float) -> int:
    window_len = int(round(window_sec * fs))
    step = int(round(window_len * (1 - overlap)))
    return max(0, (signal_len - window_len) // step + 1)


def split_signal_by_time(signal: np.ndarray, ratios: dict) -> dict:
    """把同一段錄音依時間先後、不重疊地切成 train/val/test 三段。

    這是避免 data leakage 的關鍵：因為這批資料每個 (condition, severity, load)
    只有一個檔案，不能整個檔案分去某一個 split，只能在檔案內部按時間切開，
    確保切窗時，train 的窗跟 test 的窗來自時間上完全不重疊的區段。

    ratios 例如 {"train": 0.7, "val": 0.15, "test": 0.15}
    """
    n = signal.shape[0]
    order = ["train", "val", "test"]
    cuts = {}
    start = 0
    for i, split in enumerate(order):
        if i == len(order) - 1:
            end = n
        else:
            end = start + int(round(n * ratios[split]))
        cuts[split] = signal[start:end]
        start = end
    return cuts
