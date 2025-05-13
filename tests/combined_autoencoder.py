import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import re

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


class CombinedDataset(Dataset):
    def __init__(self, dem_vectors, landuse_vectors, soil_vectors, 
                 air_vectors, ep_vectors, prec_vectors, rof_vectors):
        # 确保所有向量按相同顺序排列
        self.basin_ids = dem_vectors["baisn_id"]
        
        # 分类型归一化
        self.normalize_vectors = {
            'dem': self._normalize(dem_vectors["encoder_vector"]),
            'landuse': self._normalize(landuse_vectors["encoder_vector"]),
            'soil': self._normalize(soil_vectors["encoder_vector"]),
            'air': self._normalize(air_vectors["encoder_vector"]),
            'ep': self._normalize(ep_vectors["encoder_vector"]),
            'prec': self._normalize(prec_vectors["encoder_vector"]),
            'rof': self._normalize(rof_vectors["encoder_vector"])
        }
        
    def _normalize(self, vectors):
        """对单类向量进行归一化"""
        vectors = np.vstack(vectors)
        # mean = vectors.mean(axis=0)
        # std = vectors.std(axis=0)
        # return (vectors - mean) / (std + 1e-8)
        min_val = vectors.min(axis=0)
        max_val = vectors.max(axis=0)
        return (vectors - min_val) / (max_val - min_val + 1e-8)
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.basin_ids)
        
    def __getitem__(self, idx):
        return {
            "basin_id": self.basin_ids[idx],
            "dem": torch.FloatTensor(self.normalize_vectors['dem'][idx]),
            "landuse": torch.FloatTensor(self.normalize_vectors['landuse'][idx]),
            "soil": torch.FloatTensor(self.normalize_vectors['soil'][idx]),
            "air": torch.FloatTensor(self.normalize_vectors['air'][idx]),
            "ep": torch.FloatTensor(self.normalize_vectors['ep'][idx]),
            "prec": torch.FloatTensor(self.normalize_vectors['prec'][idx]),
            "rof": torch.FloatTensor(self.normalize_vectors['rof'][idx])
        }

class CombinedAutoencoder(nn.Module):
    def __init__(self):
        super(CombinedAutoencoder, self).__init__()
        # 空间特征统一层 (20->32)
        self.space_unify = nn.Linear(20, 70)
        # 时间特征统一层 (50->32)
        self.time_unify = nn.Linear(150, 70)
                
        # 后续编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(70*7, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 解码器保持输出224维(32*7)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 70*7)
        )
        
        # 添加还原层
        self.dem_restore = nn.Linear(70, 20)
        self.landuse_restore = nn.Linear(70, 20)
        self.soil_restore = nn.Linear(70, 20)
        self.air_restore = nn.Linear(70, 150)
        self.ep_restore = nn.Linear(70, 150)
        self.prec_restore = nn.Linear(70, 150)
        self.rof_restore = nn.Linear(70, 150)
        
    def forward(self, dem, landuse, soil, air, ep, prec, rof):
        # 统一维度
        dem = self.space_unify(dem)
        landuse = self.space_unify(landuse)
        soil = self.space_unify(soil)
        air = self.time_unify(air)
        ep = self.time_unify(ep)
        prec = self.time_unify(prec)
        rof = self.time_unify(rof)
        
        # 拼接特征
        x = torch.cat([dem, landuse, soil, air, ep, prec, rof], dim=1)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # 将224维输出拆分为7个32维特征
        decoded = decoded.view(-1, 7, 70)
        
        # 分别还原各特征到原始维度
        dem_restored = self.dem_restore(decoded[:, 0])
        landuse_restored = self.landuse_restore(decoded[:, 1])
        soil_restored = self.soil_restore(decoded[:, 2])
        air_restored = self.air_restore(decoded[:, 3])
        ep_restored = self.ep_restore(decoded[:, 4])
        prec_restored = self.prec_restore(decoded[:, 5])
        rof_restored = self.rof_restore(decoded[:, 6])
        
        # 拼接还原后的特征作为最终输出
        restored = torch.cat([
            dem_restored, landuse_restored, soil_restored,
            air_restored, ep_restored, prec_restored, rof_restored
        ], dim=1)
        return encoded, restored

def train_autoencoder(dataset, batch_size=32, epochs=100, learning_rate=0.001, test_size=0.2):
    # 划分训练集和测试集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[split:], indices[:split]
    
    # 创建子集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = CombinedAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch_list = []
    train_losses = []
    test_losses = []

    # 训练模型并计算测试集loss
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            # 获取各特征
            dem = batch["dem"]
            landuse = batch["landuse"]
            soil = batch["soil"]
            air = batch["air"]
            ep = batch["ep"]
            prec = batch["prec"]
            rof = batch["rof"]
            
            # 前向传播
            encoded, decoded = model(dem, landuse, soil, air, ep, prec, rof)
            
            # 拼接特征作为目标
            target = torch.cat([dem, landuse, soil, air, ep, prec, rof], dim=1)
            loss = criterion(decoded, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        epoch_list.append(epoch+1)
        train_losses.append(train_loss/len(train_loader))
        
        # 计算测试集loss
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                dem = batch["dem"]
                landuse = batch["landuse"]
                soil = batch["soil"]
                air = batch["air"]
                ep = batch["ep"]
                prec = batch["prec"]
                rof = batch["rof"]
                
                encoded, decoded = model(dem, landuse, soil, air, ep, prec, rof)
                target = torch.cat([dem, landuse, soil, air, ep, prec, rof], dim=1)
                test_loss += criterion(decoded, target).item()
            test_losses.append(test_loss/len(test_loader))
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
    
    # 在函数末尾，返回模型之前添加以下代码
    output_dir = Path(__file__).parent.parent / "outputs_FC"
    output_dir.mkdir(exist_ok=True)
    
    # 保存训练损失
    train_loss_df = pd.DataFrame({
        'epoch': epoch_list,
        'train_loss': train_losses
    })
    train_loss_path = output_dir / "train_losses.csv"
    train_loss_df.to_csv(train_loss_path, index=False)
    
    # 保存测试损失
    test_loss_df = pd.DataFrame({
        'epoch': epoch_list,
        'test_loss': test_losses
    })
    test_loss_path = output_dir / "test_losses.csv"
    test_loss_df.to_csv(test_loss_path, index=False)
    
    print(f"训练损失已保存到: {train_loss_path}")
    print(f"测试损失已保存到: {test_loss_path}")
    
    return model

def get_combined_vectors():
    # 从model_FC.py中获取各向量
    dem_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "dem_encoder_output.csv"
    landuse_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "landuse_encoder_output.csv"
    soil_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "soil_encoder_output.csv"
    air_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Air_encoder_output.csv"
    ep_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Ep_encoder_output.csv"
    prec_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "Prec_encoder_output.csv"
    rof_csv_path = Path(__file__).parent.parent / "outputs" / "Encoder_output" / "ROF_encoder_output.csv"
    
    # 读取各向量数据
    dem_vectors = read_dem_vectors(dem_csv_path)
    landuse_vectors = read_landuse_vectors(landuse_csv_path)
    soil_vectors = read_landuse_vectors(soil_csv_path)
    air_vectors = read_timeserious_vectors(air_csv_path)
    ep_vectors = read_timeserious_vectors(ep_csv_path)
    prec_vectors = read_timeserious_vectors(prec_csv_path)
    rof_vectors = read_timeserious_vectors(rof_csv_path)
    
    return dem_vectors, landuse_vectors, soil_vectors, air_vectors, ep_vectors, prec_vectors, rof_vectors

def load_and_inference(dataset, model_path):
    """加载训练好的模型并进行推理"""
    model = CombinedAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_encoded = []
    all_basin_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取各特征
            dem = batch["dem"]
            landuse = batch["landuse"]
            soil = batch["soil"]
            air = batch["air"]
            ep = batch["ep"]
            prec = batch["prec"]
            rof = batch["rof"]
            
            # 前向传播只获取编码结果
            encoded, _ = model(dem, landuse, soil, air, ep, prec, rof)
            
            all_encoded.append(encoded.numpy())
            all_basin_ids.extend(batch["basin_id"].numpy())
    
    # 合并所有batch的结果
    all_encoded = np.vstack(all_encoded)
    return all_basin_ids, all_encoded

def save_encoded_vectors(basin_ids, encoded_vectors, output_path):
    """保存编码结果到CSV"""
    df = pd.DataFrame(encoded_vectors)
    df.insert(0, "basin_id", basin_ids)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # 获取所有向量数据
    all_vectors = get_combined_vectors()
    
    # 创建数据集
    dataset = CombinedDataset(*all_vectors)

    # 训练自编码器
    model = train_autoencoder(dataset, batch_size = 2, epochs = 200, learning_rate = 0.0001)
    
    # 保存模型到outputs_FC文件夹
    output_dir = Path(__file__).parent.parent / "outputs_FC"
    output_dir.mkdir(exist_ok=True)  # 确保目录存在
    model_path = output_dir / "combined_autoencoder.pth"
    torch.save(model.state_dict(), model_path)

    # 进行推理并保存结果
    basin_ids, encoded_vectors = load_and_inference(dataset, model_path)

    encoder_output_path = Path(__file__).parent.parent / "outputs_FC" / "encoded_output_vectors.csv"
    save_encoded_vectors(basin_ids, encoded_vectors, encoder_output_path)
    print(f"编码结果已保存到: {encoder_output_path}")