import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import gc
import random
import rasterio
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from rasterio import features
from rasterio.windows import Window

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
                    surface_data = src_1.read(1, window=window_1, boundless=True, fill_value=src_1.nodata)
                    # surface_data = torch.from_numpy(surface_data).long()  # 转换为torch.Tensor并确保为整数类型
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
                        "surface_array": surface_data,
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
    max_cols = max(item["cols"] for item in extracted_data)+1

    # 获取所有dem_array的最大/最小值
    min_dem = min(np.min(item["surface_array"]) for item in extracted_data if item["surface_array"].size > 0)
    max_dem = max(np.max(item["surface_array"]) for item in extracted_data if item["surface_array"].size > 0)
    # 获取所有subbasin_array的最小值
    min_subbasin = min(np.min(item["subbasin_array"]) for item in extracted_data if item["subbasin_array"].size > 0)
    max_subbasin = max(np.max(item["subbasin_array"]) for item in extracted_data if item["subbasin_array"].size > 0)

    for item in extracted_data:
        # 处理dem_array
        original_dem = item ["surface_array"]
        pad_rows = max_rows - original_dem.shape[0]
        pad_cols = max_cols - original_dem.shape[1]
        item["surface_array"] = np.pad(
            original_dem,
            pad_width=((0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=min_dem
        )
        reshape_dem = item["surface_array"]
        # 标准化公式
        normalized_dem = ((reshape_dem - min_dem) / (max_dem - min_dem)).astype(np.float32)
        item["surface_array"] = normalized_dem


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
    print("扩展后的 surface_array 维度:", extracted_data[0]["surface_array"].shape)
    print("扩展后的 subbasin_array 维度:", extracted_data[0]["subbasin_array"].shape)
    print("扩展后的 surface_array 维度:", extracted_data[1]["surface_array"].shape)
    print("扩展后的 subbasin_array 维度:", extracted_data[1]["subbasin_array"].shape)

    return extracted_data


def get_cnn_input_lists(data):
    """
        将每个字典中的 dem_array 和 subbasin_array 融合为 cnn_input
        """
    results = []
    for item in data:
        # 提取数组并立即删除原始引用（减少内存占用）
        surface_value = item.pop("surface_array").astype(np.float32)  # 转换为 float32 节省内存
        subbasin = item.pop("subbasin_array").astype(np.float32)

        # 合并为多通道数组（假设沿第三维度堆叠，如 CNN 的通道）
        cnn = np.stack([surface_value, subbasin], axis=-1)  # 形状 (rows, cols, 2)
        # 强制释放内存
        del surface_value, subbasin
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

        # 调整维度顺序 [N_frames, N_frames, channel] → [channel, N_frames, N_frames]
        data = data.transpose(2, 0, 1)  # 或 data.permute(2, 0, 1)（若数据是 PyTorch 张量）

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

# class CorrDataSet(Dataset):
#
#     """To use CorrDataSet:
#
#     ds = CorrDataSet(data_file)
#     dataloader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
#     for sample_batched in dataloader:
#         do_something(sample)
#     """
#
#     def __init__(self, data_file):
#         self.data_dict = torch.load(data_file)
#         self.uids = list(self.data_dict.keys())
#         np.random.shuffle(self.uids)
#
#     def __len__(self):
#         return len(self.uids)
#
#     def __getitem__(self, idx):
#
#         uid = self.uids[idx]
#         raw_data_path = self.data_dict[uid]["data"]
#         target_path = self.data_dict[uid]["target"]
#         raw_data = torch.load(raw_data_path)
#         target_data = torch.load(target_path)
#         sample = {"data": raw_data, "target": target_data}
#
#         return sample


# def one_time(Y):
#     """
#     Calculates one-time correlation function
#     from a two-time correlation function
#
#     Parameters
#     ----------
#     Y : torch tensor
#         a two-time correlation function. Shape is (N_roi, N_frames, N_frames).
#
#     Returns
#     -------
#     res : torch tensor
#         one-time correlation function, excluding delay=0. Shape is (N_roi, N_frames-1).
#
#     """
#     bs = Y.shape[0]
#     step = Y.shape[-1]
#     res = torch.zeros(bs, step - 1)
#     for i in range(1, step):
#         res[:, i - 1] = (
#             torch.diagonal(Y, offset=i, dim1=2, dim2=3).mean(axis=2).view(bs)
#         )
#     return res


def one_time(Y):
    """
    Calculates one-time correlation function from a two-time correlation function.
    Y shape: (batch_size, N_roi, N_frames, N_frames)
    """
    bs, N_roi, step, _ = Y.shape
    res = torch.zeros(bs, N_roi, step - 1)  # 保留 N_roi 维度

    for i in range(1, step):
        # 提取对角线: shape [batch_size, N_roi, step - i]
        diag = torch.diagonal(Y, offset=i, dim1=2, dim2=3)
        # 计算均值: shape [batch_size, N_roi]
        diag_mean = diag.mean(dim=2)
        res[:, :, i - 1] = diag_mean

    # 调整形状: [batch_size, N_roi, step-1] → [batch_size, (step-1)*N_roi]
    res = res.reshape(bs, -1)
    return res

# def double_cost(Y_out, Y_t):
#     """
#     Calculates the cost function, which includes both the MSE(2TCF) and MSE(1TCF)
#
#     Parameters
#     ----------
#     Y_out : torch tensor
#         Model output. Shape is (batch_size, N_frames, N_frames).
#     Y_t : torch tensor
#         Target. Shape is (batch_size, N_frames, N_frames).
#
#     Returns
#     -------
#     cost function
#
#     """
#     return torch.mean((Y_out - Y_t) ** 2) + torch.mean(
#         (one_time(Y_out) - one_time(Y_t)) ** 2
#     )

def double_cost(Y_out, Y_t):
    # 确保输入形状一致
    assert Y_out.shape == Y_t.shape, "Y_out 和 Y_t 形状不一致"

    # 计算 2TCF 的 MSE
    mse_2tcf = torch.mean((Y_out - Y_t)  **  2)

    # 计算 1TCF 的 MSE
    ot_out = one_time(Y_out)
    ot_tgt = one_time(Y_t)
    mse_1tcf = torch.mean((ot_out - ot_tgt)  **  2)

    return mse_2tcf + mse_1tcf

def one_time_cost(Y_out, Y_t):
    """ Returns the MSE of 1TCF between the model output Y_out and the target Y_t. """

    return torch.mean((one_time(Y_out) - one_time(Y_t)) ** 2)


def setup_nn(
    dataloader,
    latent_space_dimn=2,
    lr=0.001,
    savefile=None,
    device="cpu",
    weight_decay=0,
    ch1=10,
    ch2=10,
    k=1,
):
    """
    Initialize the model and all its attributes needed for training.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        loader for the training dataset.
    latent_space_dimn : int, optional
        The default is 2.
    lr : int, optional
        learning rate. The default is 0.001.
    savefile : str , optional
        name of the file to load the model from. The default is None.
    device : str, optional
        'cpu' or 'cuda'. The default is 'cpu'.
    weight_decay : float, optional
        regularization parameter for Adam optimized. The default is 0.
    ch1 : int, optional
        number of channels in the first hidden layer of encoder. The default is 10.
    ch2 : int, optional
        number of channels in the second hidden layer of encoder. The default is 10.
    k : int, optional
        kernel size. The default is 1.

    Returns
    -------
    net : AutoEncoder_2D
        The model.
    optimizer : torch.optim.Optimizer
        Optimizer algorithm.
    scheduler : torch.optim.lr_scheduler
        Scheduler for updating the learning rate.
    cost_function : function
        Cost function.

    """
    if savefile is not None:
        net = torch.load(savefile)      # 加载预训练模型
        net.eval()

    else:
        print(f"Running on {device}")
        ksize = [k, k]              # 卷积核大小（如 3 表示 3x3 的卷积核）
        channels = [2, ch1, ch2]    # ch1, ch2：编码器卷积层的通道数（如 [1, 10, 10] 表示输入通道 1，两层卷积输出通道均为 10）。
        X = next(iter(dataloader))["src_val"]
        dim_tensor = list(X.shape)  # [batch, 1, N_frames, N_frames]
        net = AutoEncoder_2D(dim_tensor, channels, ksize, latent_space_dimn).to(device)

    cost_function = double_cost
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9995)

    return net, optimizer, scheduler, cost_function


def set_seed(seed):
    """
    Fixing random seed for all libraries.
    """

    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# 构建模型
class Conv2D_params:
    """Initialize the parameters of the model.

    Parameters:
    ----------
    dimn_tensor : list
        size of the input data used for the model training.
        [batch size, number of channels, N_frames, N_frames]
    hidden_layers_list : list
        list of channels in all comvolutional layers
    ksize : int
        size of the convolutional kernel
    latent_space_dimn : int
        number of latent variables
    """

    def __init__(
            self,
            dimn_tensor=[None, None, None, None],
            hidden_layers_list=None,
            ksize=None,
            latent_space_dimn=None,
    ):
        self.batchsize = dimn_tensor[0]
        self.channels = dimn_tensor[1]
        self.nX = dimn_tensor[2]
        self.nY = dimn_tensor[3]
        self.hidden_layers_list = hidden_layers_list
        self.ksize = ksize
        self.latent_space_dimn = latent_space_dimn


class Encoder_2D(nn.Module):
    """
    Class for encoder.

    Parameters:
    ----------
    dimn_tensor : list
        size of the input data used for the model training.
        [batch size, number of channels, N_frames, N_frames]
    hidden_layers_list : list
        list of channels in all comvolutional layers
    ksize : int
        size of the convolutional kernel
    latent_space_dimn : int
        number of latent variables
    """

    def __init__(self, dimn_tensor, hidden_layers_list, ksize, latent_space_dimn):

        # Input tensors are ( batchsize , channels , nX , nY )

        super(Encoder_2D, self).__init__()

        batchsize, channels, nX, nY = dimn_tensor

        n_layers = len(hidden_layers_list) - 1

        len_signal_conv_X = nX
        len_signal_conv_Y = nY

        # set up convolutional layers
        self.f_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_layers_list[i],
                    hidden_layers_list[i + 1],
                    kernel_size=ksize[i],
                    padding=(ksize[i] - 1) // 2,
                )
                for i in range(n_layers)
            ]
        )

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        # set up linear outout layer
        self.f_linear_out = nn.Linear(
            len_signal_conv_X * len_signal_conv_Y * hidden_layers_list[-1],
            latent_space_dimn,
        )

        nn.init.xavier_uniform_(self.f_linear_out.weight)

        # Save some network parameters
        self.conv2d_params = Conv2D_params(
            dimn_tensor, hidden_layers_list, ksize, latent_space_dimn
        )

    def forward(self, x):

        # perform convolution and ReLU
        for i, conv_i in enumerate(self.f_conv):
            x = conv_i(x)
            x = F.relu(x)

        # linear transformation ot the low diminsional latent space
        batchsize, features, nX, nY = x.size()
        x = self.f_linear_out(x.reshape(batchsize, 1, features * nX * nY))

        return x


class Decoder_2D(nn.Module):
    """
    Class for decoder.

    Parameters:
    ----------
    fc_outputsize : int
        number of the parameters in the first layer of decoder
    nX, nY : int, int
        dimensions of the convolutional laers in decoder
    channels_list: list
        list of channels in each layer of the decoder
    ksize : int
        size of the convolutional kernel
    latent_dimn : int
        number of latent variables
    """

    def __init__(self, latent_dimn, fc_outputsize, nX, nY, channels_list, ksize):

        # Input tensors are ( batchsize , latent_dimn )

        super(Decoder_2D, self).__init__()

        n_layers = len(channels_list) - 1
        ksize = ksize[::-1]

        self.f_linear_in = nn.Linear(latent_dimn, fc_outputsize)

        nn.init.xavier_uniform_(self.f_linear_in.weight)

        self.f_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels_list[i],
                    channels_list[i + 1],
                    kernel_size=ksize[i],
                    padding=(ksize[i] - 1) // 2,
                )
                for i in range(n_layers)
            ]
        )

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        self.fc_outputsize = fc_outputsize

        self.channels_list = channels_list

        self.nX = nX
        self.nY = nY
        self.nx_conv = nX
        self.ny_conv = nY

    def forward(self, x):

        x = self.f_linear_in(x).reshape(
            x.size()[0], self.channels_list[0], self.nx_conv, self.ny_conv
        )

        for i, conv_i in enumerate(self.f_conv[:-1]):
            x = conv_i(x)
            x = F.relu(x)

        x = self.f_conv[-1](x)

        return x


def get_decoder2d_fcoutputsize_from_encoder2d_params(
        encoder_hidden_layers_list, ksize, nX, nY
):
    """ Calculate parameters for constructing decoder"""

    decoder_channels = encoder_hidden_layers_list[-1::-1]

    # n_layers = len( encoder_hidden_layers_list ) - 1

    len_signal_conv_X = nX
    len_signal_conv_Y = nY

    fc_outputsize = len_signal_conv_X * len_signal_conv_Y * decoder_channels[0]

    return fc_outputsize


class AutoEncoder_2D(nn.Module):
    """Combination of Encoder_2D and Decoder_2D

    Parameters:
    ----------
    dimn_tensor : list
        size of the input data used for the model training.
        [batch size, number of channels, N_frames, N_frames]
    hidden_layers_list : list
        list of channels in all comvolutional layers
    ksize : int
        size of the convolutional kernel
    latent_space_dimn : int
        number of latent variables

    """

    def __init__(self, dimn_tensor, hidden_layers_list, ksize, latent_space_dimn):
        super(AutoEncoder_2D, self).__init__()

        self.encoder = Encoder_2D(
            dimn_tensor, hidden_layers_list, ksize, latent_space_dimn
        )

        fc_outputsize = get_decoder2d_fcoutputsize_from_encoder2d_params(
            self.encoder.conv2d_params.hidden_layers_list,
            self.encoder.conv2d_params.ksize,
            self.encoder.conv2d_params.nX,
            self.encoder.conv2d_params.nY,
        )

        self.decoder = Decoder_2D(
            self.encoder.conv2d_params.latent_space_dimn,
            fc_outputsize,
            dimn_tensor[2],
            dimn_tensor[3],
            self.encoder.conv2d_params.hidden_layers_list[-1::-1],
            self.encoder.conv2d_params.ksize,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_latent_space_coordinates(self, x):
        return self.encoder(x)



# 训练器
def train_autoencoder(
    train_dataloader,
    validation_dataloader,
    net,
    optimizer,
    scheduler,
    cost_function,
    epochs,
    checkpoint_period,
    device,
    indicator="",
    batchsize=2,
    save_folder="output/",
):
    """
    Train the model

    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        Loader of the training dataset.
    validation_dataloader : torch.utils.data.DataLoader
        Loader of the validation dataset.
    net : AutoEncoder_2D
        The model.
    optimizer : torch.optim.Optimizer
        Optimizer algorithm.
    scheduler : torch.optim.lr_scheduler
        scheduler for changing the learning rate
    cost_function : function
        Cost function.
    epochs : int
        Number of epoch to training.
    checkpoint_period : int
        Period of epoch to save the model
    device : string
        'cuda' or 'cpu'
    indicator : str, optional
        Addition to the names of saved files. The default is ''.
    batchsize : int, optional
        The default is 2.
    save_folder : str, optional
        folder to save the model files. The default is 'output/'.

    Returns
    -------
    train_error : list (float)
        Cost function for training set at each epoch.
    validation_error : list (float)
        Cost function for validation set at each epoch.

    """

    train_error = []
    validation_error = []
    best_score = 1e9

    for i in range(epochs):
        train_costs = []
        for sample_batched in train_dataloader:

            cost = 0
            optimizer.zero_grad()

            X = sample_batched["src_val"].float().to(device)
            Y = sample_batched["src_val"].float().to(device)
            X_hat = net(X)
            cost = cost_function(Y, X_hat)
            cost.backward()
            optimizer.step()

            train_costs.append(cost.cpu().detach().numpy())

            del X, Y, cost
            if torch.cuda.torch.cuda.is_available():
                torch.cuda.empty_cache()

        scheduler.step()

        eps_val = test_autoencoder(
            validation_dataloader, net, cost_function, device
        )  # test on the validation set
        train_error.append(np.mean(train_costs))
        validation_error.append(eps_val)

        # update the best model
        if validation_error[i] < best_score:
            best_score = validation_error[i]
            torch.save(net, save_folder + "/autoencoder2d_best" + indicator)

        # check conditions for early stopping
        if (
            i > 8
            and np.mean(validation_error[-4:]) > np.mean(validation_error[-8:-4])
            and train_error[-1] < validation_error[-1]
        ):
            print("Saving current autoencoder model to disk")
            break

        print(
            "EPOCH = %d, COST = %.6f, validation_error = %.6f"
            % (i + 1, train_error[-1], validation_error[-1])
        )

        # save the model periodically
        if (i + 1) % checkpoint_period == 0:
            print("Saving current autoencoder model to disk")
            torch.save(net, save_folder + "/autoencoder2d_" + str(i + 1) + indicator)



    return train_error, validation_error


def test_autoencoder(test_dataloader, net, cost_function, device, savefile=None, indicator=""):
    """
    Calculate the cost function for the model on a test set

    Parameters
    ----------
    test_dataloader : torch.utils.data.DataLoader
        Loader for the test dataset.
    net : AutoEncoder_2D
        The model.
    cost_function : function
        Cost function.
    device : string
        'cuda' or 'cpu'.
    savefile : str, optional
        path to the model file if to load from the file. The default is None.
    indicator : str, optional
        Addition to the names of saved files. The default is ''.

    Returns
    -------
    float
        the mean test error.

    """

    if savefile:
        net = torch.load(savefile)
        net.eval()

    test_costs = []

    for sample_batched in test_dataloader:

        X_test = sample_batched["src_val"].float().to(device)
        Y_test = sample_batched["src_val"].float().to(device)
        Y_pred = net(X_test)
        test_cost = cost_function(Y_test, Y_pred).cpu().detach().numpy()
        test_costs.append(test_cost)

        del X_test, Y_test, Y_pred
        if torch.cuda.torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(test_costs)

if __name__ == "__main__":
    # 准备数据集
    # 1.预数据处理
    # 假设您的矢量数据保存在 'basins.shp'
    basin_file = r'G:\BasicDatasets\YBHM_Basin_sub.shp'
    surface_file = r'F:\AutoEncoderDatasets\SurfaceData\land_reclass_env.tif'
    subbasin_cell_path = r'G:\BasicDatasets\YBHM_Subbasin_5.tif'

    # 假设土地利用类型有10种，嵌入维度为16
    num_classes = 10
    embedding_dim = 16
    # num_classes = 26
    # embedding _dim = 30
    # 定义嵌入层
    embedding_layer = nn.Embedding(num_classes, embedding_dim)
    # 为了避免由于将流域边界外的dem设置为零导致神经网络将边界误判为高程差，因此考虑获取每个流域的最小外接矩形的dem数据，融合流域的唯一标识栅格数据
    # 从而即能保留流域的dem信息，又能够保留各个流域的边界属性，通过神经网络中进行更有效地提取各个流域地表特征
    extracted_data = extract_raster_by_polygons(surface_file, subbasin_cell_path, basin_file)  # 读取每个流域的栅格数据
    # process_data = process_shape_size_normalized(extracted_data)
    # 流域形状与地表类型数据（dem、landuse、soiltype）进行融合
    cnn_input = get_cnn_input_lists(extracted_data)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(cnn_input, cnn_input, test_size=0.2, random_state=42)

    basin_id_list = [d['id'] for d in cnn_input]
    cnn_input_list = [d['cnn_input'] for d in cnn_input]
    train_basin_id_list = [d['id'] for d in X_train]
    train_cnn_input_list = [d['cnn_input'] for d in X_train]
    test_basin_id_list = [d['id'] for d in X_test]
    test_cnn_input_list = [d['cnn_input'] for d in X_test]

    # 创建数据集和数据加载器
    basin_dataset = BasinDataset(basin_id_list, cnn_input_list)  # basin_dataset维度（31，1352，1963，2）
    train_dataset = BasinDataset(train_basin_id_list, train_cnn_input_list)
    test_dataset = BasinDataset(test_basin_id_list, test_cnn_input_list)
    print(basin_dataset)
    batch_size = 4
    basin_loader = DataLoader(basin_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=basin_dataset.generate_batch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=train_dataset.generate_batch, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.generate_batch)


    #
    # # define parameters
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # train_data_file = "logs/dataset_training"
    # # validation_data_file = "logs/dataset_validation"
    #
    # epochs = 60
    # checkpoint_period = 60
    # savefile = None
    # lr = 0.00001
    # # batch_size = 8
    # latent_space_dimn = 20
    # ch1 = 10
    # ch2 = 10
    # k = 1
    # wd = 0
    # seed = 0
    #
    # save_folder = "../output/"
    # if not os.path.isdir(save_folder):
    #     os.makedirs(save_folder)  # 创建文件夹（包括父目录）
    #     print(f"已创建文件夹 '{save_folder}'")
    # else:
    #     print(f"文件夹 '{save_folder}' 已存在")
    # indicator = f"_lr_{lr}_latent_space_{latent_space_dimn}_batchsize_{batch_size}_cv_{ch1}_{ch2}_k_{k}_{k}_wd_{wd}"
    #
    # set_seed(seed)
    #
    # # load the dataset
    # # train_ds = CorrDataSet(train_data_file)
    # # val_ds = CorrDataSet(validation_data_file)
    #
    # # train_dataloader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=0)
    # # validation_dataloader = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=0)
    #
    # # initialize everythig for the training
    # model, optimizer, scheduler, cost_function = setup_nn(train_loader, latent_space_dimn, lr, savefile, device, wd, ch1, ch2, k)
    #
    # # train the model
    # T, V = train_autoencoder(
    #     train_loader,
    #     test_loader,
    #     model,
    #     optimizer,
    #     scheduler,
    #     cost_function,
    #     epochs,
    #     checkpoint_period,
    #     device,
    #     indicator,
    #     batch_size,
    #     save_folder,
    # )
    #
    # print("train_loss = " ,T, "val_loss = ", V)
    #
    # torch.save(T, save_folder + "/train_error" + indicator)
    # torch.save(V, save_folder + "/validation_error" + indicator)