"""
讀取原始檔案 (.mat / .tdms) -> numpy array

需要套件（本機/雲端執行環境請先安裝，沙盒環境沒有網路無法安裝）：
    pip install scipy nptdms numpy pandas

備註：這幾個檔案很大（振動一個檔 ~245MB, 電流溫度一個檔 ~300MB），
建議用 mmap / 逐段讀取，不要一次把 45 個檔案全部載進記憶體。
本檔案提供的函式都是「一次載入一個檔案」，在 build_dataset.py 裡逐檔處理完就釋放。
"""

import numpy as np

try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    from nptdms import TdmsFile
except ImportError:
    TdmsFile = None


def load_vibration_mat(path) -> np.ndarray:
    """回傳 shape = (n_samples, 4) 的振動訊號，欄位順序：
    [x_direction_housing_A, y_direction_housing_A, x_direction_housing_B, y_direction_housing_B]

    實際 .mat 格式（Siemens Test.Lab 輸出）為巢狀 struct：
        Signal.y_values.values  -> (N, 4) float64，4 個通道依序對應上述欄位
    """
    if sio is None:
        raise ImportError("需要 scipy: pip install scipy")

    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    # --- 格式 1：巢狀 struct（Siemens Test.Lab .mat，這批資料實際格式）---
    if "Signal" in mat:
        sig = mat["Signal"]
        try:
            data = np.asarray(sig.y_values.values).astype(np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            if data.shape[1] < 4:
                raise ValueError(f"{path.name}: 振動通道數不足 4（實際 {data.shape[1]}）")
            return data[:, :4]
        except AttributeError:
            raise KeyError(
                f"{path.name}: Signal struct 格式非預期，"
                f"欄位: {getattr(sig, '_fieldnames', '?')}"
            )

    # --- 格式 2：頂層具名欄位（Mendeley README 描述的標準格式，作為 fallback）---
    cols = [
        "x_direction_housing_A",
        "y_direction_housing_A",
        "x_direction_housing_B",
        "y_direction_housing_B",
    ]
    channels = []
    for c in cols:
        if c not in mat:
            raise KeyError(
                f"{path.name} 裡沒有欄位 '{c}'，實際欄位有: "
                f"{[k for k in mat.keys() if not k.startswith('__')]}"
            )
        channels.append(np.asarray(mat[c]).squeeze().astype(np.float32))
    return np.stack(channels, axis=1)  # (N, 4)


def load_acoustic_mat(path) -> np.ndarray:
    """回傳 shape = (n_samples,) 的聲音訊號 (單位 Pa)。

    實際 .mat 格式（Siemens Test.Lab 輸出）為巢狀 struct：
        Signal.y_values.values -> (N,) float64
    """
    if sio is None:
        raise ImportError("需要 scipy: pip install scipy")

    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    # --- 格式 1：巢狀 struct（Siemens Test.Lab .mat）---
    if "Signal" in mat:
        sig = mat["Signal"]
        try:
            return np.asarray(sig.y_values.values).squeeze().astype(np.float32)
        except AttributeError:
            raise KeyError(
                f"{path.name}: Signal struct 格式非預期，"
                f"欄位: {getattr(sig, '_fieldnames', '?')}"
            )

    # --- 格式 2：頂層 'values' 欄位（舊版格式，fallback）---
    if "values" not in mat:
        raise KeyError(
            f"{path.name} 裡沒有欄位 'values'，實際欄位有: "
            f"{[k for k in mat.keys() if not k.startswith('__')]}"
        )
    return np.asarray(mat["values"]).squeeze().astype(np.float32)


def load_current_temp_tdms(path) -> dict:
    """回傳 dict，包含：
        temperature_A, temperature_B : (N_temp,) 攝氏
        current_U, current_V, current_W : (N_cur,) 安培
        fs : 這個檔案實際量到的取樣率 (Hz)，用檔案裡的 wf_increment 算出來，
             不要假設是 config.py 裡寫的 CURRENT_TEMP_FS

    ⚠️ 這批資料的 channel 名稱是硬體通道代號（例如
    'cDAQ9185-1F486B5Mod1/ai0'），不包含 temperature/current 這種字，
    原本用名稱關鍵字比對的寫法完全比對不到，已經改成：
      1. 用 unit_string 分組（'°C' -> 溫度, 'A' -> 電流）
      2. 同一組內依 channel 名稱排序，依序指定 A/B 或 U/V/W

    這是照你實際跑 inspect_tdms.py 印出來的結果寫的：
        Mod1/ai0 (°C), Mod1/ai1 (°C)              -> temperature_A, temperature_B
        Mod2/ai0 (A), Mod2/ai2 (A), Mod2/ai3 (A)  -> current_U, current_V, current_W

    ⚠️ 缺相處理：scan_tdms_health.py 掃過全部 45 個檔案後確認，9 個 BPFO 檔案
    （3 個負載 × 3 個嚴重度）的 V/W 相長度是 0，只有 U 相有資料。
    所以這裡不再強制要求三相都有資料，改成：
      - current_U 一定要有（沒有就報錯）
      - current_V / current_W 若為空，回傳長度 0 的陣列，並在
        available_phases 標明實際有幾相可用
    前處理流程只會用到 U 相（見 features_current.py 開頭的說明）。
    """
    if TdmsFile is None:
        raise ImportError("需要 nptdms: pip install nptdms")

    tdms = TdmsFile.read(str(path))

    temp_channels = []
    current_channels = []
    fs = None

    for group in tdms.groups():
        for channel in group.channels():
            props = channel.properties
            unit = props.get("unit_string", "")
            increment = props.get("wf_increment")
            if increment and fs is None:
                fs = 1.0 / increment

            if unit in ("degC", "°C", "C"):
                temp_channels.append(channel)
            elif unit in ("A", "Amp", "Ampere"):
                current_channels.append(channel)
            # 其他 unit（例如 Voltage）先不管，這批資料只用得到溫度跟電流

    temp_channels.sort(key=lambda ch: ch.name)
    current_channels.sort(key=lambda ch: ch.name)

    if len(temp_channels) != 2:
        raise KeyError(
            f"{path.name}: 預期 2 個溫度 channel（unit=degC），實際找到 {len(temp_channels)} 個: "
            f"{[ch.name for ch in temp_channels]}。先跑 `python inspect_tdms.py {path}` 看實際狀況。"
        )
    if len(current_channels) != 3:
        raise KeyError(
            f"{path.name}: 預期 3 個電流 channel（unit=A），實際找到 {len(current_channels)} 個: "
            f"{[ch.name for ch in current_channels]}。先跑 `python inspect_tdms.py {path}` 看實際狀況。"
        )

    # 讀取三相，允許 V/W 為空（BPFO 檔案就是這種狀況）
    phases = [ch[:].astype(np.float32) for ch in current_channels]
    available_phases = sum(1 for p in phases if len(p) > 0)

    if len(phases[0]) == 0:
        raise ValueError(
            f"{path.name}: U 相電流是空的（長度 0），這個檔案無法用於電流分析。"
            f"三相長度分別為 {[len(p) for p in phases]}。"
        )

    result = {
        "temperature_A": temp_channels[0][:].astype(np.float32),
        "temperature_B": temp_channels[1][:].astype(np.float32),
        "current_U": phases[0],
        "current_V": phases[1],
        "current_W": phases[2],
        "available_phases": available_phases,
        "fs": fs,
    }
    return result


def load_current_temp_tdms_with_time(path) -> dict:
    """跟 load_current_temp_tdms 一樣，但額外附上每個 channel 的時間軸（若有 wf_increment/wf_start_offset）。
    溫度和電流常常不同取樣率，切窗前務必先確認彼此的實際 sampling rate，
    不要直接假設都是 config.py 裡的 CURRENT_TEMP_FS。
    """
    if TdmsFile is None:
        raise ImportError("需要 nptdms: pip install nptdms")
    tdms = TdmsFile.read(str(path))
    out = {}
    for group in tdms.groups():
        for channel in group.channels():
            props = channel.properties
            increment = props.get("wf_increment")  # 每點間隔秒數
            fs = (1.0 / increment) if increment else None
            out[channel.name] = {
                "data": channel[:].astype(np.float32),
                "fs": fs,
                "unit": props.get("unit_string", ""),
            }
    return out