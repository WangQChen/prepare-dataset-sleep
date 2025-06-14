import re
import pandas as pd
from collections import defaultdict

RML_URLS_PATH = "./download_urls_list.txt"
INDEX_PATH = "./task_index.csv"

def generate_task_index():
    edf_count_map = defaultdict(set)

    # 1. 扫描 txt 文件
    with open(RML_URLS_PATH, 'r') as f:
        lines = [line.strip() for line in f]

    for line in lines:
        # 匹配 EDF 分块
        edf_match = re.search(r'/V3/APNEA_EDF/(\d{8}-\d+)/\1\[(\d{3})\]\.edf', line)
        if edf_match:
            pid, part = edf_match.group(1), edf_match.group(2)
            edf_count_map[pid].add(part)

    # 2. 生成表格
    data = []
    for pid in sorted(edf_count_map.keys()):
        sep_num = len(edf_count_map[pid])
        data.append({'id': pid, 'sep_num': sep_num, 'done': False})

    df = pd.DataFrame(data)
    df.to_csv(INDEX_PATH, index=False)
    print(f"✅ 已保存任务索引文件到 {INDEX_PATH}")

if __name__ == "__main__":
    generate_task_index()