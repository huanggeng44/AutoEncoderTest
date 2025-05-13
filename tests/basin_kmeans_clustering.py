import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

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
    features = df.iloc[:, 1:].values  # 跳过第一列(basin_id)
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(features)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'basin_id': basin_ids,
        'cluster': clusters
    })

    print(result_df)
    
    # 确定输出路径
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果
    output_path = output_dir / 'basin_cluster_results_4.csv'
    result_df.to_csv(output_path, index=False)
    
    print(f"聚类完成，结果已保存到: {output_path}")
    return result_df

if __name__ == "__main__":
    # 输入文件路径
    input_csv = r"G:\Projects\Test\my-project\outputs_FC\encoded_output_vectors.csv"
    
    # 输出目录
    output_dir = r"G:\Projects\Test\my-project\outputs_FC"
    
    # 执行聚类
    cluster_results = cluster_basin_vectors(input_csv, n_clusters=4, output_dir=output_dir)