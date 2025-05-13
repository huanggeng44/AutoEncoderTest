import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Dict

def parse_tensor_value(tensor_str: str) -> float:
    """Parse single tensor value to float"""
    pattern = r"tensor\(([\d\.eE+-]+)(?:,\s*device='cuda:\d+')?\)"
    match = re.search(pattern, tensor_str)
    if not match:
        raise ValueError(f"Invalid tensor value format: {tensor_str[:50]}...")
    return float(match.group(1))

def read_landuse_vectors() -> Dict[float, np.ndarray]:
    """读取土地利用向量数据"""
    # 构建相对路径
    # csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "landuse_encoder_output.csv"
    csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "soil_encoder_output.csv"
    
    df = pd.read_csv(csv_path)
    landuse_vectors = {}
    
    for _, row in df.iterrows():
        basin_id = float(row['basin_id'])
        # 提取所有feature列的值
        features = []
        for col in df.columns:
            if col.startswith('feature_'):
                features.append(parse_tensor_value(row[col]))
        landuse_vectors[basin_id] = np.array(features, dtype=np.float32)
    
    return landuse_vectors

if __name__ == "__main__":
    # 使用示例
    landuse_data = read_landuse_vectors()
    
    # 打印前3个结果验证
    for i, (basin_id, vector) in enumerate(landuse_data.items()):
        if i >= 3:
            break
        print(f"流域ID: {basin_id}")
        print(f"土地利用向量: {vector}")
        print(f"向量形状: {vector.shape}\n")