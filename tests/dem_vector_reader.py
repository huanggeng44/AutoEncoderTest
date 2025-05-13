import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Dict

def parse_tensor_string(tensor_str: str) -> np.ndarray:
    """Parse tensor string to numpy array"""
    # More robust pattern that handles various number formats and whitespace
    pattern = r"tensor\(\[([\d\s\.,eE+-]+)\](?:,\s*device='cuda:\d+')?\)"
    match = re.search(pattern, tensor_str)
    if not match:
        raise ValueError(f"Invalid tensor string format: {tensor_str[:50]}...")
    
    # Clean and parse values
    values = match.group(1).replace("\n", "").replace(" ", "")
    return np.fromstring(values, sep=",", dtype=np.float32)

def read_dem_vectors() -> Dict[float, np.ndarray]:
    """读取DEM向量数据"""
    # 构建相对路径
    csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "dem_encoder_output.csv"
    
    df = pd.read_csv(csv_path)
    dem_vectors = {}
    
    for _, row in df.iterrows():
        basin_id = float(row['basin_id'])
        vector = parse_tensor_string(row['feature_0'])
        dem_vectors[basin_id] = vector
    
    return dem_vectors

if __name__ == "__main__":
    # 使用示例
    dem_data = read_dem_vectors()
    
    # 打印前3个结果验证
    for i, (basin_id, vector) in enumerate(dem_data.items()):
        if i >= 3:
            break
        print(f"流域ID: {basin_id}")
        print(f"DEM向量: {vector}")
        print(f"向量形状: {vector.shape}\n")