import pandas as pd
import re
import numpy as np
from pathlib import Path
from typing import Dict
import torch

# DEM
def dem_parse_tensor_string(tensor_str: str) -> np.ndarray:
    """Parse tensor string to numpy array"""
    # More robust pattern that handles various number formats and whitespace
    pattern = r"tensor\(\[([\d\s\.,eE+-]+)\](?:,\s*device='cuda:\d+')?\)"
    match = re.search(pattern, tensor_str)
    if not match:
        raise ValueError(f"Invalid tensor string format: {tensor_str[:50]}...")

    # Clean and parse values
    values = match.group(1).replace("\n", "").replace(" ", "")
    return np.fromstring(values, sep=",", dtype=np.float32)


def read_dem_vectors(csv_path) -> Dict[float, np.ndarray]:
    """读取DEM向量数据"""

    df = pd.read_csv(csv_path)
    dem_vectors = {}

    for _, row in df.iterrows():
        basin_id = float(row['basin_id'])
        vector = dem_parse_tensor_string(row['feature_0'])
        dem_vectors[basin_id] = vector

    dem_basin_id_list = []
    dem_encoder_vectors_list = []
    dem_encoder_vector_sizes = []
    # 打印前3个结果验证
    for i, (dem_basin_id, dem_vector) in enumerate(dem_vectors.items()):
        dem_basin_id_list.append(dem_basin_id)
        dem_encoder_vectors_list.append(dem_vector)
        dem_encoder_vector_sizes.append(dem_vector.shape)
        # print(f"流域ID: {dem_basin_id}")
        # print(f"DEM向量: {dem_vector}")
        # print(f"向量形状: {dem_vector.shape}\n")

    dem_encoder_vector_dic = {
        "baisn_id": dem_basin_id_list,
        "encoder_vector": dem_encoder_vectors_list,
        "vector_size": dem_encoder_vector_sizes
    }

    return dem_encoder_vector_dic


# Landuse & Soil
def landuse_parse_tensor_value(tensor_str: str) -> float:
    """Parse single tensor value to float"""
    pattern = r"tensor\(([\d\.eE+-]+)(?:,\s*device='cuda:\d+')?\)"
    match = re.search(pattern, tensor_str)
    if not match:
        raise ValueError(f"Invalid tensor value format: {tensor_str[:50]}...")
    return float(match.group(1))


def read_landuse_vectors(csv_path) -> Dict[float, np.ndarray]:
    """读取土地利用向量数据"""
    # 构建相对路径

    df = pd.read_csv(csv_path)
    landuse_vectors = {}

    for _, row in df.iterrows():
        basin_id = float(row['basin_id'])
        # 提取所有feature列的值
        features = []
        for col in df.columns:
            if col.startswith('feature_'):
                features.append(landuse_parse_tensor_value(row[col]))
        landuse_vectors[basin_id] = np.array(features, dtype=np.float32)

    landuse_basin_id_list = []
    landuse_encoder_vectors_list = []
    landuse_encoder_vector_sizes = []
    # 打印前3个结果验证
    for i, (landuse_basin_id, landuse_vector) in enumerate(landuse_vectors.items()):
        landuse_basin_id_list.append(landuse_basin_id)
        landuse_encoder_vectors_list.append(landuse_vector)
        landuse_encoder_vector_sizes.append(landuse_vector.shape)
        # print(f"流域ID: {landuse_basin_id}")
        # print(f"DEM向量: {landuse_vector}")
        # print(f"向量形状: {landuse_vector.shape}\n")

    landuse_encoder_vector_dic = {
        "baisn_id": landuse_basin_id_list,
        "encoder_vector": landuse_encoder_vectors_list,
        "vector_size": landuse_encoder_vector_sizes
    }
    return landuse_encoder_vector_dic


# Timeserious
def timeserious_parse_tensor_value(tensor_str: str) -> float:
    """Parse tensor value string to float"""
    pattern = r"tensor\(([\d\.eE+-]+)\)"
    match = re.search(pattern, tensor_str)
    if not match:
        raise ValueError(f"Invalid tensor value format: {tensor_str[:50]}...")
    return float(match.group(1))


def read_timeserious_vectors(csv_path) -> Dict[float, np.ndarray]:
    df = pd.read_csv(csv_path)
    timeserious_vectors = {}

    for _, row in df.iterrows():
        basin_id = float(row['basin_id'])
        # 提取所有feature列的值
        features = []
        for col in df.columns:
            if col.startswith('feature_'):
                features.append(timeserious_parse_tensor_value(row[col]))
        timeserious_vectors[basin_id] = np.array(features, dtype=np.float32)

    timeserious_basin_id_list = []
    timeserious_encoder_vectors_list = []
    timeserious_encoder_vector_sizes = []
    # 打印前3个结果验证
    for i, (timeserious_basin_id, timeserious_vector) in enumerate(timeserious_vectors.items()):
        timeserious_basin_id_list.append(timeserious_basin_id)
        timeserious_encoder_vectors_list.append(timeserious_vector)
        timeserious_encoder_vector_sizes.append(timeserious_vector.shape)
        # print(f"流域ID: {timeserious_basin_id}")
        # print(f"DEM向量: {timeserious_vector}")
        # print(f"向量形状: {timeserious_vector.shape}\n")

    timeserious_encoder_vector_dic = {
        "baisn_id": timeserious_basin_id_list,
        "encoder_vector": timeserious_encoder_vectors_list,
        "vector_size": timeserious_encoder_vector_sizes
    }
    return timeserious_encoder_vector_dic

def get_datasets_encoder_vector_dic():
    # 构建相对路径
    dem_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "dem_encoder_output.csv"
    dem_vector = read_dem_vectors(dem_csv_path)
    print(dem_vector)

    landuse_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "landuse_encoder_output.csv"
    landuse_vector = read_landuse_vectors(landuse_csv_path)
    print(landuse_vector)

    soil_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "soil_encoder_output.csv"
    soil_vector = read_landuse_vectors(soil_csv_path)
    print(soil_vector)

    Air_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Air_encoder_output.csv"
    air_vector = read_timeserious_vectors(Air_csv_path)
    print(air_vector)

    Ep_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Ep_encoder_output.csv"
    ep_vector = read_timeserious_vectors(Ep_csv_path)
    print(ep_vector)

    Prec_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Prec_encoder_output.csv"
    prec_vector = read_timeserious_vectors(Prec_csv_path)
    print(prec_vector)

    ROf_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "ROF_encoder_output.csv"
    rof_vector = read_timeserious_vectors(ROf_csv_path)
    print(rof_vector)

    return dem_vector, landuse_vector, soil_vector, air_vector, ep_vector, prec_vector, rof_vector


if __name__ == "__main__":

    dem_vector, landuse_vector, soil_vector, air_vector, ep_vector, prec_vector, rof_vector = get_datasets_encoder_vector_dic()


