import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import re

def extract_tensor_value(x):
    """从tensor(xxx)字符串中提取数值"""
    match = re.search(r'tensor\(([-+]?\d*\.\d+)\)', str(x))
    return float(match.group(1)) if match else 0.0

def cluster_basin_vectors(csv_path, n_clusters=6, output_dir=None):
    """
    读取流域向量CSV文件并进行K-means聚类
    
    参数:
        csv_path: 输入CSV文件路径
        n_clusters: 聚类数量，默认为6
        output_dir: 输出目录，默认为None（与输入文件同目录）
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 提取流域ID和特征向量
    basin_ids = df['basin_id'].values
    features = df.iloc[:, 1:].applymap(extract_tensor_value).values
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(features)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'basin_id': basin_ids,
        'cluster': clusters
    })
    
    # 确定输出路径
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    output_path = output_dir / 'ep_cluster_results.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"聚类完成，结果已保存到: {output_path}")
    return result_df

if __name__ == "__main__":
    # 输入文件路径
    # input_csv = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Air_encoder_output.csv"
    # input_csv = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Ep_encoder_output.csv"
    input_csv = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Prec_encoder_output.csv"
    # input_csv = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "ROF_encoder_output.csv"
    
    # 输出目录
    output_dir = r"G:\Projects\Test\my-project\outputs_clusting"
    
    # 执行聚类
    cluster_results = cluster_basin_vectors(input_csv, n_clusters=4, output_dir=output_dir)