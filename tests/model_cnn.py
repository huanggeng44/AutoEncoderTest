
import os

os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'
from osgeo import gdal,osr,ogr
import geopandas as gpd
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import gc
from rasterio import features

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import PackedSequence
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask



def extract_raster_by_polygons(dem_path, subbasin_cell_path, shapefile_path):
    """
    提取每个矢量面对应的栅格数据及其行列数
    参数:
        dem_path: DEM栅格文件路径
        shapefile_path: 流域矢量面文件路径
    返回:
        list: 每个元素为字典，包含原始面ID、栅格数组、行数、列数
    """
    results = []

    # 读取矢量面数据
    gdf = gpd.read_file(shapefile_path)
    print(f"发现 {len(gdf)} 个流域面")

    # 打开DEM数据
    with rasterio.open(dem_path) as src_1:
        with rasterio.open(subbasin_cell_path) as src_2:
            # 获取栅格基本信息
            crs_1 = src_1.crs
            crs_2 = src_2.crs
            transform_1 = src_1.transform
            transform_2 = src_2.transform
            out_shape_2 = src_2.shape
            print(f"栅格CRS: {crs_1}, 分辨率: {src_1.res}")
            print(f"栅格CRS: {crs_2}, 分辨率: {src_2.res}")
            print(f"矢量CRS: {gdf.crs}")

            # 检查坐标系一致性
            if gdf.crs != crs_1:
                print(f"警告: 矢量({gdf.crs})与栅格({crs_1})坐标系不一致！")
                print("正在进行动态坐标转换...")
                gdf = gdf.to_crs(crs_1)
            elif gdf.crs != crs_2:
                print(f"警告: 矢量({gdf.crs})与栅格({crs_2})坐标系不一致！")
                print("正在进行动态坐标转换...")
                gdf = gdf.to_crs(crs_2)

            # 遍历每个面
            for idx, row in gdf.iterrows():
                geom = row.geometry
                hybas_id = row['HYBAS_ID']  # 假设属性表中包含HYBAS_ID字段

                try:
                    # 获取面的地理边界
                    minx, miny, maxx, maxy = geom.bounds

                    # 地理坐标转像素坐标
                    # 左上角坐标 (minx, maxy)
                    row_start_1, col_start_1 = src_1.index(minx, maxy)
                    row_start_2, col_start_2 = src_2.index(minx, maxy)
                    # 右下角坐标 (maxx, miny)
                    row_end_1, col_end_1 = src_1.index(maxx, miny)
                    row_end_2, col_end_2 = src_2.index(maxx, miny)

                    # 计算行列数
                    height_1 = row_end_1 - row_start_1
                    width_1 = col_end_1 - col_start_1
                    height_2 = row_end_2 - row_start_2
                    width_2 = col_end_2 - col_start_2

                    # 创建读取窗口
                    window_1 = Window(col_start_1, row_start_1, width_1, height_1)
                    window_2 = Window(col_start_2, row_start_2, width_2, height_2)

                    # 读取栅格数据 (自动处理边界外情况)
                    dem_data = src_1.read(1, window=window_1, boundless=True, fill_value=src_1.nodata)
                    # subbasin_data = src_2.read(1, window=window_2, boundless=False, fill_value=0)
                    transform = src_2.window_transform(window_2)

                    # 生成二值掩膜（面内1，面外0）
                    binary_subbasin_data = features.rasterize(
                        [(geom, 1)],
                        out_shape=(height_2, width_2),
                        transform=transform,
                        fill=0,
                        dtype=np.uint8
                    )

                    # 记录结果
                    results.append({
                        "hybas_id": hybas_id,
                        "dem_array": dem_data,
                        "subbasin_array": binary_subbasin_data,
                        "rows": height_1,
                        "cols": width_1,
                        "window_bounds": (minx, miny, maxx, maxy)
                    })

                    print(f"面 {hybas_id} 提取成功 | 尺寸: {height_1}行×{width_1}列")

                except Exception as e:
                    print(f"面 {hybas_id} 提取失败: {str(e)}")
                    continue

    return results

def process_shape_size_normalized(extracted_data):
    """
    将每个流域的数据进行统一维度
    并且标准化流域的dem数据
    :param extracted_data: extract_raster_by_polygons()返回的字典数据。
            "hybas_id"：各个流域id
            "dem_array": 各个流域(env)的dem
            "subbasin_array": 由矢量数据转为面栅格数据的流域唯一值
            "rows": height: 流域图像有多少行像元
            "cols": width: 流域图像有多少列像元
            "window_bounds": (minx, miny, maxx, maxy)

    :return:
    """
    print(type(extracted_data))  # 应输出 <class 'list'>
    print(type(extracted_data[0]))  # 应输出 <class 'dict'>
    print(extracted_data[0].keys())  # 应输出字典的键，如 ['hybas_id', 'dem_array', ...]

    # 获取最大行列数
    max_rows = max(item["rows"] for item in extracted_data)
    max_cols = max(item["cols"] for item in extracted_data)

    # 获取所有dem_array的最大/最小值
    min_dem = min(np.min(item["dem_array"]) for item in extracted_data if item["dem_array"].size > 0)
    max_dem = max(np.max(item["dem_array"]) for item in extracted_data if item["dem_array"].size > 0)
    # 获取所有subbasin_array的最小值
    min_subbasin = min(np.min(item["subbasin_array"]) for item in extracted_data if item["subbasin_array"].size > 0)
    max_subbasin = max(np.max(item["subbasin_array"]) for item in extracted_data if item["subbasin_array"].size > 0)

    for item in extracted_data:
        # 处理dem_array
        original_dem = item ["dem_array"]
        pad_rows = max_rows - original_dem.shape[0]
        pad_cols = max_cols - original_dem.shape[1]
        item["dem_array"] = np.pad(
            original_dem,
            pad_width=((0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=min_dem
        )
        reshape_dem = item["dem_array"]
        # 标准化公式
        normalized_dem = ((reshape_dem - min_dem) / (max_dem - min_dem)).astype(np.float32)
        item["dem_array"] = normalized_dem


        # 处理subbasin_array（逻辑相同）
        original_subbasin = item["subbasin_array"]
        pad_rows = max_rows - original_subbasin.shape[0]
        pad_cols = max_cols - original_subbasin.shape[1]
        item["subbasin_array"] = np.pad(
            original_subbasin,
            pad_width=((0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=min_subbasin
        )
        # reshape_subbasin = item["subbasin_array"]
        # # 标准化公式
        # normalized_subbasin = ((reshape_subbasin - min_subbasin) / (max_subbasin - min_subbasin)).astype(np.float32)
        # item["subbasin_array"] = normalized_subbasin

    # 验证结果
    # 验证标准化后全局范围
    # normalized_values_min = min(np.min(item["dem_array"]) for item in extracted_data if item["dem_array"].size > 0)
    # normalized_values_max = max(np.max(item["dem_array"]) for item in extracted_data if item["dem_array"].size > 0)
    # print(f"Min-Max标准化范围：[{normalized_values_min:.2f}, {np.max(normalized_values_max):.2f}]")
    print("扩展后的 dem_array 维度:", extracted_data[0]["dem_array"].shape)
    print("扩展后的 subbasin_array 维度:", extracted_data[0]["subbasin_array"].shape)
    print("扩展后的 dem_array 维度:", extracted_data[1]["dem_array"].shape)
    print("扩展后的 subbasin_array 维度:", extracted_data[1]["subbasin_array"].shape)

    return extracted_data


def get_cnn_input_lists(data):
    """
        将每个字典中的 dem_array 和 subbasin_array 融合为 cnn_input
        """
    results = []
    for item in data:
        # 提取数组并立即删除原始引用（减少内存占用）
        dem = item.pop("dem_array").astype(np.float32)  # 转换为 float32 节省内存
        subbasin = item.pop("subbasin_array").astype(np.float32)

        # 合并为多通道数组（假设沿第三维度堆叠，如 CNN 的通道）
        cnn = np.stack([dem, subbasin], axis=-1)  # 形状 (rows, cols, 2)
        # 强制释放内存
        del dem, subbasin
        gc.collect()

        results.append({
            "id": item.pop("hybas_id"),
            "cnn_input": cnn
        })

    return results


# 1.数据预处理
def process_shapefile(gdf):
    runoff_data = []
    runoff_columns = []
    for i in range(1, 515):  # 514个月
        col_name = f'band_{i}'
        if col_name in gdf.columns:
            runoff_data.append(gdf[col_name].values)
            runoff_columns.append(col_name)
        else:
            raise ValueError(f"Field {col_name} not found in shapefile.")
    # basin_ids = np.array(gdf['Id'])
    basin_ids = np.array(gdf['HYBAS_ID'])
    runoff_data = np.array(runoff_data).T  # 转置，使每一行代表一个时间步
    # print(runoff_data)

    return basin_ids, runoff_data, runoff_columns,

def preprocess_data(id_list, dataframe, feature_cols):
    scaler = MinMaxScaler()
    # 设置显示的最大行数和列数
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    dataframe[feature_cols] = scaler.fit_transform(dataframe[feature_cols])

    # print(dataframe)
    # runoff_data = dataframe[feature_cols].values
    # return runoff_data

    # 将行索引重置为普通列（默认列名为 'index'）
    dataframe.index = range(1, len(dataframe) + 1)
    df_reset = dataframe.reset_index()

    # 将 'index' 列重命名为 'basin_id'
    df_reset.rename(columns={'index': 'basin_id'}, inplace=True)
    # 使用 loc 进行替换
    df_reset.loc[:, 'basin_id'] = id_list
    # print(df_reset)
    return df_reset


# class ChunkedDataset(Dataset):
#     def __init__(self, dataset,chunk_axis=0, chunk_size=4):
#         """
#             chunk_axis: 分块维度 (0:样本维度，2/3:空间维度)
#             chunk_size: 每个块的大小
#         """
#         self.dataset = dataset
#         self.chunk_axis = chunk_axis
#         self.chunk_size = chunk_size
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         # 按需加载单个样本
#         item = self.dataset[idx]  # 获取字典类型的样本
#         # 正确提取并转换数值
#         src_val = item["src_val"].astype(np.float32)  # 先转换类型
#         return {
#             "src_ids": torch.tensor(item["src_ids"]),
#             "src_val": torch.from_numpy(src_val)
#         }

# 创建时间序列数据集
class BasinDataset(Dataset):
    def __init__(self, basin_id, data):
        self.basin_id = basin_id
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        basin_id = self.basin_id[idx]
        data = self.data[idx]

        item = {
            "src_ids": basin_id,
            "src_val": data
        }

        return item

    def generate_batch(self, item_list):
        src_tgt = [(x["src_ids"], x["src_val"]) for x in item_list]
        # src_tgt = sorted(src_tgt, key=lambda x: len(x[0]), reverse=True)
        # 分别提取源和目标的 ID、runoff时间序列到单独的列表中
        src_ids = [x[0] for x in src_tgt]
        src_val = [x[1] for x in src_tgt]
        # 确保所有子数组形状一致
        assert all(arr.shape == src_val[0].shape for arr in src_val), "数组形状不一致"
        # 将列表转换为3D/4D NumPy数组
        src_val_array = np.array(src_val)

        # 将处理好的数据存储在一个字典 batch 中，并将其转换为 PyTorch 张量（Tensor）格式。
        batch = {
            "src_ids": src_ids,
            "src_val": torch.from_numpy(src_val_array).to(torch.float32)
            # "src_lengths": src_lengths,
        }
        return batch

class ShapeChecker(nn.Module):
    def forward(self, x):
        print(x.shape)  # 打印各层输出尺寸
        return x

# 2.构建Seq2Seq模型
class CNNEncoder(nn.Module):
    def __init__(self, input_channels=2):
        super(CNNEncoder, self).__init__()
        # 编码器结构
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            # ShapeChecker(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # ShapeChecker(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # ShapeChecker(),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.encoder(x)


class SizeAdjust(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        # 双线性插值对齐尺寸
        return F.interpolate(x,
                             size=self.target_size,
                             mode="bilinear",
                             align_corners=False)

class CNNDecoder(nn.Module):
    def __init__(self, output_channels=2):
        super(CNNDecoder, self).__init__()

        # 解码器结构
        self.decoder = nn.Sequential(
            # Block 1
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # ShapeChecker(),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 2
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # ShapeChecker(),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3
            nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # ShapeChecker(),
            nn.Sigmoid()  # 输出在0-1之间
        )
        self.decoder.add_module("size_adjust", SizeAdjust((1352, 1964)))


    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, input_channels=2):
        super(Autoencoder, self).__init__()
        self.encoder = CNNEncoder(input_channels)
        self.decoder = CNNDecoder(input_channels)

    def forward(self, x):
        # 维度转换 (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        # 转换回原始维度 (N, C, H, W) -> (N, H, W, C)
        reconstructed = reconstructed.permute(0, 2, 3, 1)
        return reconstructed


# 3.模型训练
def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    # scaler = GradScaler()
    epoch_loss = 0
    for batch in tqdm(train_dataloader):
    # for batch in train_dataloader:
        # 数据准备
        inputs = batch["src_val"].to(device)

        with autocast(device_type='cuda'):
            # outputs = model(inputs)
            # loss = criterion(outputs, inputs)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, inputs)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 记录损失
            epoch_loss += loss.item()
            train_dataloader.set_postfix({"loss": f"{loss.item():.4f}"})
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()

        # # 强制释放未使用的内存
        # gc.collect()
        # torch.cuda.empty_cache()  # 如果使用GPU
    return epoch_loss / len(train_dataloader)

def test(model, test_dataloader, criterion, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch["src_val"].to(device)
            targets = batch["src_val"].to(device)
            outputs = model(inputs,targets)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    average_loss = total_loss / len(test_dataloader)
    return average_loss

def get_encoder_output(model, data_loader, device):
    encoder_batch_tensors = []
    basin_id_list = []
    with torch.no_grad():
        for batch in data_loader:
            src = batch["src_val"].to(device)
            src_id = batch["src_ids"]
            out, hidden, cell = model.encoder(src)
            encoder_batch_tensors.append(out)
            basin_id_list.append(src_id)

        # 列表中的tensor沿着维度0（行）拼接
        out_val = torch.cat(encoder_batch_tensors, dim=0)
        basin_id_list = [item for sublist in basin_id_list for item in sublist]
        encoder_output_dict = {
            "basin_id": basin_id_list,
            "encoder_output": out_val
        }
        # print("Concatenated along dim=0:\n", encoder_output)

    return encoder_output_dict

def cluster_Kmeans(encoder_output, n_clusters, random_state):

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    # encoder聚类
    kmeans.fit(encoder_output["encoder_output"])
    # PCA聚类
    # kmeans.fit(encoder_output["pca_output"])

    # 获取聚类标签
    labels = kmeans.labels_
    # （可选）获取聚类中心
    cluster_centers = kmeans.cluster_centers_
    # 创建一个包含 ID 和对应聚类标签的 DataFrame
    cluster_output = pd.DataFrame({
        'basin_id': encoder_output["basin_id"],
        'cluster': labels
    })

    return cluster_output, cluster_centers

def PCA_process(id_data, timeseries_data):
    # 创建PCA对象，不指定主成分数量
    n_components = 0.95
    pca_auto = PCA(n_components = n_components)

    # 拟合并转换数据
    X_pca_auto = pca_auto.fit_transform(timeseries_data)

    print(f"降维后的维度：{X_pca_auto.shape[1]}")

    # 将结果转换为DataFrame
    columns_name = [f'PC{i + 1}' for i in range(X_pca_auto.shape[1])]
    pca_auto_df = pd.DataFrame(X_pca_auto, columns=columns_name)
    pca_auto_df['basin_id'] = id_data
    print(pca_auto_df.head())

    # 保存结果
    outputs_dir = f"../outputs/pca_outputs/"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    pca_save_name = f"{now_time}_PCA_{n_components}_output.csv"
    pca_output_file = os.path.join(outputs_dir, pca_save_name)
    pca_auto_df.to_csv(pca_output_file, index=False)

    pca_output_dict = {
        "basin_id": id_data,
        "pca_output": X_pca_auto.tolist()
    }

    # 查看每个主成分的方差比例
    explained_variance = pca_auto.explained_variance_ratio_
    print("各主成分的方差比例：", explained_variance)
    # 查看累计方差比例
    cumulative_variance = np.cumsum(explained_variance)
    print("累计方差比例：", cumulative_variance)

    # 将结果转换为DataFrame
    if explained_variance.shape[0] == cumulative_variance.shape[0]:
        columns = [f"PC_{i}" for i in range(1, explained_variance.shape[0]+1)]
        # 为行命名
        row_names = ['explained_variance', 'cumulative_variance']
        # 将两个数组堆叠成一个二维数组，形状为 (2, 38)
        data = np.vstack((explained_variance, cumulative_variance))
        # 创建DataFrame，并指定列名称
        pca_var_df = pd.DataFrame(data, columns=columns, index=row_names)
    else:
        print("两者长度不同。")

    # 保存结果
    pca_var_name = f"{now_time}_PCA_var_{n_components}_output.csv"
    pca_var_output_file = os.path.join(outputs_dir, pca_var_name)
    pca_var_df.to_csv(pca_var_output_file, index=False)


    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题
    # 可视化累计方差比例（可选）
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('主成分数量')
    plt.ylabel('累计方差比例')
    plt.title('累计方差比例 vs 主成分数量')
    plt.grid(True)
    plt.show()

    return pca_output_dict



def main():
    # 1.预数据处理
    # 假设您的矢量数据保存在 'basins.shp'
    basin_file = r'G:\BasicDatasets\YBHM_Basin_sub.shp'
    dem_file = r'G:\BasicDatasets\YBHM_DEM_env_5.tif'
    subbasin_cell_path = r'G:\BasicDatasets\YBHM_Subbasin_5.tif'
    # 为了避免由于将流域边界外的dem设置为零导致神经网络将边界误判为高程差，因此考虑获取每个流域的最小外接矩形的dem数据，融合流域的唯一标识栅格数据
    # 从而即能保留流域的dem信息，又能够保留各个流域的边界属性，通过神经网络中进行更有效地提取各个流域地表特征
    extracted_data = extract_raster_by_polygons(dem_file,subbasin_cell_path,  basin_file)   # 读取每个流域的栅格数据
    process_data = process_shape_size_normalized(extracted_data)

    cnn_input = get_cnn_input_lists(process_data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(cnn_input, cnn_input, test_size=0.2, random_state=42)

    basin_id_list =  [d['id'] for d in cnn_input]
    cnn_input_list = [d['cnn_input'] for d in cnn_input]
    train_basin_id_list = [d['id'] for d in X_train]
    train_cnn_input_list = [d['cnn_input'] for d in X_train]
    test_basin_id_list = [d['id'] for d in X_test]
    test_cnn_input_list = [d['cnn_input'] for d in X_test]

    # 创建数据集和数据加载器
    basin_dataset = BasinDataset(basin_id_list, cnn_input_list)     # basin_dataset维度（31，1352，1963，2）
    train_dataset = BasinDataset(train_basin_id_list, train_cnn_input_list)
    test_dataset = BasinDataset(test_basin_id_list, test_cnn_input_list)
    print(basin_dataset)
    batch_size = 4
    basin_loader = DataLoader(basin_dataset, batch_size=batch_size, shuffle=False, collate_fn=basin_dataset.generate_batch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.generate_batch, num_workers=4,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.generate_batch)
    #
    #
    # 2.构建Seq2Seq模型
    # 3.模型训练
    # batch_size = 31
    # device = torch.device('cuda')
    device = torch.device('cpu')
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    # # 训练循环
    optimizer = torch.optim.Adam(model.parameters())
    N_EPOCHS = 100
    # for epoch in tqdm(range(N_EPOCHS)):
    epoch_list = []
    train_loss_list = []
    for epoch in tqdm(range(N_EPOCHS)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')

    # 以折线图形式输出epoch的loss值
    plt.plot(epoch_list, train_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.show()

    # 保存训练好的模型
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs_cnn_dem/{now_time}/"
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)  # 创建文件夹（包括父目录）
        print(f"已创建文件夹 '{outputs_dir}'")
    else:
        print(f"文件夹 '{outputs_dir}' 已存在")
    model_save_name = f"model_step_{N_EPOCHS}_ppl_{train_loss}.pth"
    model_save_path = os.path.join(outputs_dir, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    # also save the optimizers' state
    torch.save(optimizer.state_dict(), model_save_path + '.optim')

    # # 加载完成训练后的model
    # model_trained = Seq2Seq(device, embeddings, hiddens, n_layers).to(device)
    # model_test_path = "../outputs_cnn/20250311-155917/model_step_1000_ppl_0.0017708948580548167.pth"
    # model_trained.load_state_dict(torch.load(model_test_path))
    # model_trained.to(device)

    # # 测试
    # criterion = nn.MSELoss()
    # test_loss = test(model_trained, test_loader, criterion, device)
    # print(f'test_Loss: {test_loss:.4f}')

    # # 获取encoder的output
    # encoder_output = get_encoder_output(model_trained, basin_loader, device)
    # print(encoder_output)

    # # 主成分分析
    # pca_auto_output = PCA_process(basin_id_list, runoff_timeserious_list)

    # # 将encoder输出向量进行K-Means算法聚类
    # n_clusters = 8
    # random_state = 42
    # outputs_dir = f"../outputs/cluster_outputs/"
    # if not os.path.exists(outputs_dir):
    #     os.makedirs(outputs_dir)
    # now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # encoder聚类
    # cluster_output, cluster_centers = cluster_Kmeans(encoder_output, n_clusters, random_state)
    # # cluster_save_name = f"{now_time}_{n_clusters}class_output.csv"
    # cluster_save_name = f"{now_time}_GBTM_{n_clusters}class_output.csv"
    # cluster_output_file = os.path.join(outputs_dir, cluster_save_name)
    # cluster_output.to_csv(cluster_output_file, index=False)

    # PCA聚类
    # cluster_output, cluster_centers = cluster_Kmeans(pca_auto_output, n_clusters, random_state)
    # cluster_save_name = f"{now_time}_pca_{n_clusters}class_output.csv"
    # cluster_output_file = os.path.join(outputs_dir, cluster_save_name)
    # cluster_output.to_csv(cluster_output_file, index=False)
    #
    # print("聚类标签:", cluster_output)
    # print("聚类中心:\n", cluster_centers)


if __name__ == "__main__":
    main()