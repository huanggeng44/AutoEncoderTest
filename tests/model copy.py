import geopandas as gpd
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import csv


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import PackedSequence
# 1.数据预处理
def process_shapefile(gdf):
    runoff_data = []
    runoff_columns = []

    # 有多少个月数据字段
    # 定义匹配模式（精确匹配 band_ 后跟数字）
    pattern = r'^band_\d+$'  # 匹配如 band_1, band_123 等
    # 查找匹配列名
    matched_columns = [col for col in gdf.columns if re.match(pattern, col)]
    month_counts = len(matched_columns)
    print(month_counts)

    for i in range(1, month_counts + 1):  # 514个月
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
            "basin_id": basin_id,
            "runoff_val": data,
        }

        return item
        # x = self.data[idx:idx+self.seq_length]
        # y = self.data[idx+self.seq_length]
        # return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def generate_batch(self, item_list):
        src_tgt = [(x["basin_id"], x["runoff_val"]) for x in item_list]
        # src_tgt = sorted(src_tgt, key=lambda x: len(x[0]), reverse=True)
        # 分别提取源和目标的 ID、runoff时间序列到单独的列表中
        src_ids = [x[0] for x in src_tgt]
        src_val = [x[1] for x in src_tgt]
        # 计算源和目标runoff时间序列的长度
        src_lengths = [len(x) for x in src_val]

        # 将处理好的数据存储在一个字典 batch 中，并将其转换为 PyTorch 张量（Tensor）格式。
        batch = {
            "src_ids": src_ids,
            "src_val": torch.tensor(src_val, dtype=torch.float32),
            "src_lengths": src_lengths,
        }
        return batch


# 2.构建Seq2Seq模型
class Encoder(nn.Module):
    def __init__(self, device, embeddings=300, hidden_size=600, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embeddings
        self.lstm = nn.LSTM(embeddings, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embeddings)
        self.tanh = nn.Tanh()

        # init weights
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.02)
        nn.init.xavier_uniform_(self.linear.weight.data, gain=0.25)

    def forward(self, x):
        # 移除 PackedSequence 的使用
        lstm_out, (hidden, cell) = self.lstm(x)

        # linear input is the lstm output of the last word
        lineared = self.linear(lstm_out)
        out = self.tanh(lineared)

        return out, hidden, cell

class Decoder(nn.Module):
    def __init__(self, device, embedding_size=300, hidden_size=900, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embedding_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        # init parameter
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.02)
        nn.init.xavier_uniform_(self.linear.weight.data, gain=0.25)

    def forward(self, x, hidden_in, cell_in):
        # 移除 PackedSequence 的使用
        lstm_out, (hidden, cell) = self.lstm(x, (hidden_in, cell_in))

        # prediction: [seq length, batch size, embeddings]
        prediction = self.relu(self.linear(lstm_out))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, device, input_size, hidden_size, n_layers):
        super(Seq2Seq, self).__init__()
        self.device = device
        # 使用现有的 Encoder 和 Decoder 类
        self.encoder = Encoder(device, input_size, hidden_size, n_layers)
        self.decoder = Decoder(device, input_size, hidden_size, n_layers)

    def forward(self, x, y):
        # 编码器前向传播
        encoder_out, hidden, cell = self.encoder(x)
        # 解码器前向传播
        output, _, _ = self.decoder(encoder_out, hidden, cell)
        return output


# 3.模型训练
def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    batch_losses = []
    # for batch in tqdm(train_dataloader):
    for batch in train_dataloader:
        src = batch["src_val"].to(device)
        trg = batch["src_val"].to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        epoch_loss += batch_loss
        batch_losses.append(batch_loss)  # 保存当前批次的损失
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    return avg_epoch_loss, batch_losses

def test(model, test_dataloader, criterion, device):
    model.eval()

    total_loss = 0
    batch_losses = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = batch["src_val"].to(device)
            targets = batch["src_val"].to(device)
            outputs = model(inputs,targets)

            loss = criterion(outputs, targets)
            batch_loss = loss.item()
            batch_losses.append(batch_loss)  # 保存当前批次的损失
            total_loss += batch_loss

    average_loss = total_loss / len(test_dataloader)
    return average_loss, batch_losses

def get_encoder_output(model, data_loader, device,save_path=None):
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

        if save_path:
            # 创建一个DataFrame来保存结果
            num_features = encoder_output_dict["encoder_output"].shape[1]
            columns = ["basin_id"] + [f"feature_{i}" for i in range(num_features)]
            data = []
            for i in range(len(encoder_output_dict["basin_id"])):
                row = [encoder_output_dict["basin_id"][i]] + list(encoder_output_dict["encoder_output"][i])
                data.append(row)
            df = pd.DataFrame(data, columns=columns)
            # 保存为CSV文件
            df.to_csv(save_path, index=False)
            print(f"Encoder输出已保存到 {save_path}")

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
    # basin_file = r'F:\111_3\FLDAS_NOAH01_1982_2022\ROf_8201_2410_01\Sub_ROf_mm_01.shp'
    # basin_file = r'F:\111_3\FLDAS_NOAH01_1982_2022\ROf_8201_2410_01\Sub_ROf_mm_01_1_GBTM.shp'   # 采用与GBTM相同的子流域，对径流数据进行子流域平均值计算，得到的流域平均径流面数据
    basin_timeserious_type = "Ep"
    basin_timeserious_type_name = "潜在蒸散发量"
    basin_timeserious_unit = "mm"
    basin_file = f'F:\AutoEncoderDatasets\TimeseriousData\{basin_timeserious_type}_{basin_timeserious_unit}_sub_ave.shp'

    gdf = gpd.read_file(basin_file)
    basin_ids, runoff_value, feature_cols = process_shapefile(gdf)

    # 转换为DataFrame
    runoff_df = pd.DataFrame(runoff_value, columns=feature_cols)
    # 归一化处理（可选，但推荐）
    runoff_normalized = preprocess_data(basin_ids, runoff_df, feature_cols)
    print(runoff_normalized)


    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(runoff_normalized, runoff_normalized, test_size=0.2, random_state=42)

    basin_id_list = runoff_normalized['basin_id'].tolist()
    runoff_timeserious_columns = runoff_normalized.drop(columns=['basin_id'])
    runoff_timeserious_list = runoff_timeserious_columns.values.tolist()

    basin_id_list_train = X_train['basin_id'].tolist()
    runoff_timeserious_columns_train = X_train.drop(columns=['basin_id'])
    runoff_timeserious_list_train = runoff_timeserious_columns_train.values.tolist()

    basin_id_list_test = X_test['basin_id'].tolist()
    runoff_timeserious_columns_test = X_test.drop(columns=['basin_id'])
    runoff_timeserious_list_test = runoff_timeserious_columns_test.values.tolist()


    # 创建数据集和数据加载器
    basin_dataset = BasinDataset(basin_id_list, runoff_timeserious_list)
    train_dataset = BasinDataset(basin_id_list_train, runoff_timeserious_list_train)
    test_dataset = BasinDataset(basin_id_list_test, runoff_timeserious_list_test)
    # print(train_dataset[0])
    batch_size = 4
    basin_loader = DataLoader(basin_dataset, batch_size=batch_size, shuffle=False, collate_fn=basin_dataset.generate_batch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_dataset.generate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.generate_batch)


    # 2.构建Seq2Seq模型
    # 3.模型训练
    # device = torch.device('cuda')
    device = torch.device('cpu')
    input_size = len(runoff_timeserious_list[0])    # 表示输入序列的特征维度，即每个时间步的特征数量，这里等于T 。
    hidden_size = input_size    # 隐藏层的维度，通常用于表示编码器对输入序列的编码表示。
    n_layers = 1    # 循环神经网络的层数。

    # if there is an existing model, load the existing model from file
    # model_fname = "./models/_seq2seq_1698000846.3281412"
    model_fname = None
    model = None
    if not model_fname is None:
        print('loading model from ' + model_fname)
        model = torch.load(model_fname, map_location=device)
        print('model loaded')
    else:
        model = Seq2Seq(device, input_size, hidden_size, n_layers).to(device)


    # 训练循环
    criterion_train = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    N_EPOCHS = 200
    # 模型训练后的存储文件夹信息
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/{basin_timeserious_type}_{now_time}/"
    os.makedirs(outputs_dir)
    training_log_file = f"{basin_timeserious_type}_training_loss_log_step_{N_EPOCHS}.csv"
    training_log_file_path = os.path.join(outputs_dir, training_log_file)
    testing_log_file = f"{basin_timeserious_type}_testing_loss_log_step_{N_EPOCHS}.csv"
    testing_log_file_path = os.path.join(outputs_dir, testing_log_file)
    train_batch_loss_file = f"{basin_timeserious_type}_training_batch_loss_log_step_{N_EPOCHS}.csv"
    train_batch_loss_file_path = os.path.join(outputs_dir, train_batch_loss_file)
    test_batch_loss_file = f"{basin_timeserious_type}_test_batch_loss_log_step_{N_EPOCHS}.csv"
    test_batch_loss_file_path = os.path.join(outputs_dir, test_batch_loss_file)

    # 开始训练
    train_all_batch_losses = []
    train_epoch_list = []
    train_loss_list = []
    for epoch in range(N_EPOCHS):
        train_loss, batch_losses = train(model, train_loader, optimizer, criterion_train, device)
        train_epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        train_all_batch_losses.extend(batch_losses)  # 追加当前 epoch 的批次损失
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')

        # 每个 epoch 后实时保存到 CSV（避免程序中断丢失数据）
        with open(training_log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])  # 写入表头
            writer.writerows(zip(train_epoch_list, train_loss_list))

        # 保存批次级别的损失到 CSV
        with open(train_batch_loss_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'batch_loss'])  # 写入表头
            writer.writerows(enumerate(train_all_batch_losses))


    # 保存训练好的模型
    model_save_name = f"{basin_timeserious_type}_model_step_{N_EPOCHS}_ppl_{train_loss}.pth"
    model_save_path = os.path.join(outputs_dir, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    # also save the optimizers' state
    torch.save(optimizer.state_dict(), model_save_path + '.optim')

    # 加载完成训练后的model
    model_trained = Seq2Seq(device, input_size, hidden_size, n_layers).to(device)
    model_test_path = model_save_path
    model_trained.load_state_dict(torch.load(model_test_path))
    model_trained.to(device)
    #
    # 测试
    criterion_test = nn.MSELoss()
    test_all_batch_losses = []
    test_epoch_list = []
    test_loss_list = []
    for epoch in range(N_EPOCHS):
        test_loss, batch_losses = test(model_trained, test_loader, criterion_test, device)
        test_epoch_list.append(epoch)
        test_loss_list.append(test_loss)
        test_all_batch_losses.extend(batch_losses)  # 追加当前 epoch 的批次损失
        print(f'Epoch {epoch + 1}, Loss: {test_loss:.4f}')

        # 每个 epoch 后实时保存到 CSV（避免程序中断丢失数据）
        with open(testing_log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])  # 写入表头
            writer.writerows(zip(test_epoch_list, test_loss_list))
        # 保存批次级别的损失到 CSV
        with open(test_batch_loss_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['batch', 'batch_loss'])  # 写入表头
            writer.writerows(enumerate(test_all_batch_losses))


    # 获取encoder的output
    encoder_save_name = f"{basin_timeserious_type}_encoder_output.csv"
    encoder_save_path = os.path.join(outputs_dir, encoder_save_name)
    encoder_output = get_encoder_output(model_trained, basin_loader, device, save_path=encoder_save_path)
    print(encoder_output)

    # 以折线图形式输出epoch的loss值
    # 设置全局字体（中文用黑体，英文/数字用 Times New Roman）
    mpl.rcParams['font.family'] = 'sans-serif'  # 默认字体类型
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文使用黑体
    mpl.rcParams['font.serif'] = ['Times New Roman']  # 英文/数字使用 Times
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
    plt.plot(train_epoch_list, train_loss_list, marker=None, linestyle='-', color='blue', label='训练损失')
    plt.plot(test_epoch_list, test_loss_list, marker=None, linestyle='-', color='red', label='测试损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title(f'LSTM自编码器模型训练与测试损失曲线（{basin_timeserious_type_name}）')
    # 设置刻度标签字体（强制指定 Times New Roman）
    plt.xticks(fontproperties='Times New Roman', fontsize=10)
    plt.yticks(fontproperties='Times New Roman', fontsize=10)
    # 显示图例（中文）
    plt.legend(prop={'family': 'SimHei', 'size': 10})
    # 保存图像（支持中文路径）
    fig_save_path_name = f"{basin_timeserious_type}_train_loss_per_epoch.png"
    fig_save_path = os.path.join(outputs_dir, fig_save_path_name)
    plt.savefig(fig_save_path, dpi=900, bbox_inches='tight')
    plt.show()

    # # 从csv中加载loss和epoch数据进行绘图
    # epochs, losses = [], []
    # with open(training_log_file_path, 'r') as f:
    #     reader = csv.reader(f)
    #     next(reader)  # 跳过表头
    #     for row in reader:
    #         epochs.append(int(row[0]))
    #         losses.append(float(row[1]))
    #
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, losses, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curve')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('training_loss.png')  # 保存为图片
    # plt.show()

    # # 主成分分析
    # # pca_auto_output = PCA_process(basin_id_list, runoff_timeserious_list)
    #
    # # 将encoder输出向量进行K-Means算法聚类
    # n_clusters = 8
    # random_state = 42
    # outputs_dir = f"../outputs/cluster_outputs/"
    # if not os.path.exists(outputs_dir):
    #     os.makedirs(outputs_dir)
    # now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # # encoder聚类
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