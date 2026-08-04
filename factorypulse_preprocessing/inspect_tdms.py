"""
第一步一定要先跑這個腳本，看清楚你這批 tdms 檔案裡實際的 group / channel 名稱、
取樣率(wf_increment)、單位，再決定 io_utils.py 裡的欄位比對規則要不要調整。

用法：
    python inspect_tdms.py "acoustic_temp_vibration/current,temp/0Nm_Normal.tdms"

我在沙盒環境用 `strings` 對這個檔案的 header 做過粗略檢查，看到的關鍵字有：
    Temperature, NI_CjcTemperature, Current, Voltage
但看不到完整的 group/channel 樹狀結構（沒有 nptdms 可以跑），
所以 io_utils.py 裡的欄位比對是「保守的模糊比對」，第一次跑一定要用這支腳本核對過。
"""

import sys
from nptdms import TdmsFile


def main(path: str):
    tdms = TdmsFile.read(path)
    for group in tdms.groups():
        print(f"[Group] {group.name}")
        for channel in group.channels():
            props = channel.properties
            n = len(channel)
            fs = None
            if "wf_increment" in props:
                fs = 1.0 / props["wf_increment"]
            print(
                f"    - channel='{channel.name}' "
                f"len={n} unit={props.get('unit_string')} "
                f"fs={fs} start={props.get('wf_start_time')}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python inspect_tdms.py <path_to.tdms>")
        sys.exit(1)
    main(sys.argv[1])
