import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path
import os
import glob

def read_nc_file(nc_file):
    # 打开NetCDF文件
    ds = xr.open_dataset(nc_file)

    # 查看文件结构
    print(ds)
    # 查看所有维度
    print("维度:", ds.dims)

    # 查看所有变量
    print("变量:", list(ds.variables))

    # # 查看单个变量属性
    # temp = ds['temperature']
    # print("温度变量详情:", temp.attrs)

def nc_to_csv(nc_file, output_csv):
    # 读取NetCDF文件
    ds = xr.open_dataset(nc_file, decode_coords=True)

    # 确保经纬度坐标与维度正确关联
    ds = ds.set_coords(['latitude', 'longitude'])

    # 直接转换包含经纬度坐标的DataFrame
    df = ds['IWU'].to_dataframe(name='IWU').reset_index()

    # 数据透视（直接使用实际经纬度）
    df_pivot = df.pivot_table(
        index=['latitude', 'longitude'],  # 使用实际坐标列名
        columns='time',
        values='IWU'
    ).reset_index()

    # 重命名列
    df_pivot.columns = ['lat', 'lon'] + [f'IWU_{year}' for year in range(2000, 2016)]

    # 保存为CSV
    df_pivot.to_csv(output_csv, index=False)

def csv_process(input_csv, process_csv):
    # 读取CSV文件
    # df = pd.read_csv(output_csv)
    # 读取CSV时强制转换所有数值列为float类型
    df = pd.read_csv(input_csv, dtype='float',
                     converters={col: pd.to_numeric for col in pd.read_csv(input_csv, nrows=1).columns[2:]},
                     on_bad_lines='warn',
                     low_memory=False)

    # 定位需要处理的列范围（第三列到最后一列）
    columns_to_process = df.columns[2:]  # 索引2对应第三列

    # 遍历目标列，替换大于100000的值为0
    for col in columns_to_process:
        mask = df[col] > 100000
        df.loc[mask, col] = 0

    # 保存修改后的文件
    df.to_csv(process_csv, index=False)

def IND_IWU(input_csv, output_csv):
    try:
        # 读取CSV文件
        df = pd.read_csv(input_csv)

        # 显示表格基本信息（可选）
        # print("=== 原始表格基本信息 ===")
        # print(df.info())
        # print("\n=== 原始表格前几行 ===")
        # print(df.head())

        # 筛选条件：
        # lon 列的值在 (72, 99) 范围内
        # lat 列的值在 (21, 33) 范围内
        filtered_df = df[(df['lon'] >= 68) & (df['lon'] <= 99) &
                         (df['lat'] >= 6) & (df['lat'] <= 36)]

        # # 获取第一列列名
        # first_col = df.columns[0]
        # filtered_df[first_col] = filtered_df[first_col].round(2)

        # 保留原有表头
        new_columns = list(filtered_df.columns[::])
        # 设置新的表头
        filtered_df.columns = new_columns

        # # （可选）查看修改后的前几行数据
        # print("修改后的表格前几行：")
        # print(filtered_df.head())

        # 将筛选后的数据保存为新的CSV文件
        filtered_df.to_csv(output_csv, index=False)
        print(f"\n筛选后的数据已保存到 '{output_csv}'")

    except FileNotFoundError:
        print(f"文件未找到: {output_csv}")
    except pd.errors.EmptyDataError:
        print("CSV文件为空。")
    except KeyError as e:
        print(f"缺少必要的列: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def csv_to_points(csv_IND_file_path, points_IND_file_path):
    # 读取CSV文件
    csv_file = csv_IND_file_path  # 替换为你的CSV文件路径
    df = pd.read_csv(csv_file)

    # 检查CSV文件是否包含lon和lat列
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError("CSV文件必须包含'lon'和'lat'列")

    # # 修改第三列及之后的列名（取最后5个字符）# 仅irrigation数据集使用
    # new_columns = []
    # for idx, col in enumerate(df.columns):
    #     if idx >= 2:  # 第三列开始（Python从0开始计数）
    #         new_col_name = col[-5:] if len(col) >= 5 else col  # 处理不足5字符的情况
    #         new_columns.append(new_col_name)
    #     else:
    #         new_columns.append(col)
    # df.columns = new_columns
    # # 修改第三列及之后的列名（取最后5个字符）# 仅irrigation数据集使用（止）

    # 将lon和lat列转换为几何点
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # EPSG:4326表示WGS84坐标系

    # 保存为Shapefile
    gdf.to_file(points_IND_file_path, driver='ESRI Shapefile')

    print(f"Shapefile已保存到: {points_IND_file_path}")

def IND_IWU_Year(IND_shp, water_use_shp, IND_IWU_Y_csv):
    # 加载点数据（假设为WGS84坐标系，单位是度）
    points = gpd.read_file(water_use_shp)
    # 加载面数据（确保坐标系与点数据一致）
    polygon = gpd.read_file(IND_shp)
    # 对面几何体创建0.2度的缓冲区
    buffered_polygon = polygon.geometry.iloc[0].buffer(0.1)

    # 筛选位于缓冲区域内的点
    mask = points.within(buffered_polygon)
    selected_points = points[mask]
    print(f"找到 {len(selected_points)} 个有效点")

    # 定义年份列（假设列名为 Y2000-Y2015）
    year_cols = [f"IWU_{year}" for year in range(2000, 2016)]# industry数据集的年份（Y2000-Y2015）
    # year_cols = [f"Y{year}" for year in range(2011, 2019)]  # irrigation数据集的年份（IWU_ens_Y2011 - IWU_ens_Y2018）
    # 计算各年总和
    annual_totals = selected_points[year_cols].sum()
    # 生成结果DataFrame
    result_df = annual_totals.reset_index()
    result_df.columns = ['Year', 'Total_Water_Use']

    # 保存结果
    result_df.to_csv(IND_IWU_Y_csv, index=False)
# ==========================================================================
def irr_nc_to_csv(input_folder, output_folder):
    """处理NC文件并生成结构化CSV"""
    # 创建输出目录
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 遍历NC文件
    for nc_file in Path(input_folder).glob("*.nc"):
        try:
            # ===== 1. 读取和处理NC文件 =====
            ds = xr.open_dataset(nc_file)

            # ===== 2. 从文件名提取标识符 =====
            # 示例文件名格式假设为："IWU_YYYYMM.nc"
            filename = nc_file.stem  # 移除扩展名
            # 提取倒数第7位到第4位字符（假设为年份）
            year_id = filename[-4 :] if len(filename) >= 7 else "000"

            # ===== 3. 重组数据结构 =====
            # 将三维数据转换为二维 (lat*lon, time)
            df = ds["IWUens"].stack(point=("lat", "lon")).to_pandas().T  # type: ignore

            # 设置列名（格式：F{年份}_{月份}）
            new_columns = [f"F{year_id}_{int(time)}" for time in ds.time.values]  # type: ignore
            df.columns = new_columns

            # ===== 4. 添加经纬度信息 =====
            # 生成经纬度组合索引
            multi_index = ds["IWUens"].stack(point=("lat", "lon")).coords  # type: ignore
            df = df.reset_index().assign(
                lat=multi_index["lat"].values,
                lon=multi_index["lon"].values
            )[["lon", "lat"] + new_columns]  # 列顺序调整

            # ===== 5. 保存为CSV =====
            output_path = Path(output_folder) / f"{filename}.csv"
            # 使用分类写入优化大文件存储
            df.astype({col: np.float32 for col in new_columns}) \
                .to_csv(output_path, index=False)

            print(f"处理完成: {nc_file.name} -> {output_path.name}")

            # 关闭数据集释放内存
            ds.close()

        except Exception as e:
            print(f"处理失败 {nc_file.name}: {str(e)}")


def process_csv_files(folder_path, output_file):
    """
    合并2011-2018的csv文件
    :param folder_path:
    :param output_file:
    :return:
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # 存储处理后的DataFrame
    processed_data = []

    for file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(file)

        # 检查是否存在lon和lat列
        if 'lon' not in df.columns or 'lat' not in df.columns:
            print(f"文件 {os.path.basename(file)} 缺少lon或lat列，已跳过。")
            continue

        # 获取文件名（无后缀）作为新列名
        filename = os.path.splitext(os.path.basename(file))[0]

        # 确定需要求和的列（排除lon和lat）
        columns_to_sum = [col for col in df.columns if col not in ['lon', 'lat']]
        if not columns_to_sum:
            print(f"文件 {filename} 无有效列可求和，已跳过。")
            continue

        # 计算行和并生成新列
        df[filename] = df[columns_to_sum].sum(axis=1)

        # 提取所需列
        processed_df = df[['lon', 'lat', filename]]
        processed_data.append(processed_df)

    if not processed_data:
        print("没有找到可处理的有效文件。")
        return

    # 合并所有DataFrame
    merged_df = processed_data[0]
    for df in processed_data[1:]:
        merged_df = pd.merge(merged_df, df, on=['lon', 'lat'], how='outer')

    # 输出结果
    merged_df.to_csv(output_file, index=False)
    print(f"结果已保存至：{output_file}")

if __name__ == "__main__":
    # nc_file = r'F:\02WaterUse_6_man\industrial_water_use_data_codes\data\netcdf_files\Predicted_IWU_2000_to_2015.nc'
    nc_to_csv_file = r"F:\021_6\IWU_nc_to_csv.csv"
    process_csv = r"F:\021_6\IWU_process_csv.csv"
    IND_env_IWU_csv = r"F:\021_6\IWU_IND_csv.csv"
    IND_shp = r'G:\DataSets\BasinRiverData\03_country\country\India.shp'
    points_IND_file_path = r"F:\021_6\IWU_IND_points.shp"
    IND_IWU_Y_csv = r"F:\021_6\IWU_IND_Y_csv.csv"

    # read_nc_file(nc_file)
    # nc_to_csv(nc_file, nc_to_csv_file)    # nc转csv
    csv_process(nc_to_csv_file, process_csv)  # 处理数值将巨大值设置为0
    IND_IWU(process_csv, IND_env_IWU_csv)     # 提取印度国内数据
    csv_to_points(IND_env_IWU_csv, points_IND_file_path)  # 将csv文件转为shp点文件
    IND_IWU_Year(IND_shp, points_IND_file_path, IND_IWU_Y_csv)    # 计算印度逐年工业用水量

    # folder = r"F:\02WaterUse_5_ag"
    # nc_file = r'F:\02WaterUse_5_ag\IWU_ens_Y2011.nc'
    # irr_csv_folder = r"F:\022_5\IWU"
    # irr_IND_csv_folder = r"F:\022_5\IWU_IND"
    # IWU_IND_Years_file = r"F:\022_5\IWU_IND\IWU_ens_Y2011_2018.csv"
    # points_IND_file_path = r"F:\022_5\IWU_IND\IWU_ens_Y2011_2018.shp"
    # IND_IWU_Y_csv = r"F:\022_5\IWU_IND\IWU_IND_Y.csv"

    # read_nc_file(nc_file)
    # irr_nc_to_csv(folder, irr_csv_folder)
    # 提取印度用水数据
    # 创建输出目录
    # Path(irr_IND_csv_folder).mkdir(parents=True, exist_ok=True)
    # for csv_file in Path(irr_csv_folder).glob("*.csv"):
    #     filename = csv_file.stem
    #     IND_env_IWU_csv_path = Path(irr_IND_csv_folder) / f"{filename}.csv"
    #     IND_IWU(csv_file, IND_env_IWU_csv_path)

    # process_csv_files(irr_IND_csv_folder, IWU_IND_Years_file)
    # csv_to_points(IWU_IND_Years_file, points_IND_file_path)  # 将csv文件转为shp点文件
    # IND_IWU_Year(IND_shp, points_IND_file_path, IND_IWU_Y_csv)  # 计算印度逐年工业用水量


