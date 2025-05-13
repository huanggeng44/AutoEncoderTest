import os
import warnings
os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'

from osgeo import gdal,osr,ogr
import pandas as pd
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import csv
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
from shapely import wkt

def judge_file_path(file_path):
    # 获取文件所在的文件夹路径
    folder_path = os.path.dirname(file_path)
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"成功创建文件夹: {folder_path}")
        except Exception as e:
            print(f"创建文件夹失败: {e}")
    else:
        print(f"文件夹已存在: {folder_path}")

# 浏览数据
def read_data(nc_file_path):
    nc_dataset = Dataset(nc_file_path, 'r')
    # Extract the variable data
    variable_names = nc_dataset.variables.keys()
    variable_names_list = []
    print("NetCDF文件中的变量名称如下：")
    for var_name in variable_names:
        variable_names_list.append(var_name)
        print(var_name)

    # NetCDF文件中的变量名称如下：
    # cons_elec
    # lon
    # lat
    # month

    # 关闭文件
    nc_dataset.close()
    return variable_names_list

# 定义一个函数将月份数值转换为日期
def months_since_to_date(months, base_year, base_month):
    """
    将相对于基准年月（base_year, base_month）的月份数值转换为具体日期。

    参数：
    - months: 整数或数组，表示相对于基准年月的月份数。
    - base_year: 整数，基准年份。
    - base_month: 整数，基准月份（1-12）。

    返回：
    - 日期或日期数组。
    """
    dates = []
    for m in months:
        year = int(base_year + (base_month + m - 2) // 12)
        month = int((base_month + m - 2) % 12 + 1 )
        dates.append(datetime(year, month, 1))
    return dates

def read_months_info(nc_file_path):
    dataset = nc.Dataset(nc_file_path)
    months = dataset.variables['month'][:]  # 月份信息，形状为 (month,)，如果月份是索引，可以根据需要处理
    units = dataset.variables['month'].units  # 'months since 1971-1'
    base_date_str = units.split('since')[1].strip()  # '1971-1'
    base_year, base_month = map(int, base_date_str.split('-'))

    # 5. 转换所有月份数值为日期
    dates = months_since_to_date(months, base_year, base_month)
    converted_dates = []
    # 6. 输出结果
    for original_month, converted_date in zip(months, dates):
        date_str =  "F"+converted_date.strftime('%Y%m')
        converted_dates.append(date_str)
        print(f"原始月份数值: {original_month} -> 转换后的日期: {converted_date.strftime('%Y-%m-%d')}")

    return converted_dates

def nc_to_csv(nc_file_path, csv_file_path,variable_name):
    # NC文件转csv文件
    # # 打开NetCDF文件
    dataset = nc.Dataset(nc_file_path)
    # # 读取变量
    lon = dataset.variables['lon'][:]  # 经度，形状为 (grid_num,)
    lat = dataset.variables['lat'][:]  # 纬度，形状为 (grid_num,)
    water_value = dataset.variables[variable_name]  # 用水量，形状为 (month, grid_num)
    months = dataset.variables['month'][:]  # 月份信息，形状为 (month,)，如果月份是索引，可以根据需要处理

    # 获取 'months since 1971-1' 的具体信息
    months_datas = read_months_info(nc_file_path)  # 重新设置月用水量字段的字段名

    # 获取维度大小
    num_grids = len(lon)
    num_months = water_value.shape[0]

    # 创建一个空的列表，用于存储每一行的数据
    data = []

    # 遍历每个网格
    for grid in tqdm(range(num_grids)):
        row = {
            'grid_num': grid,
            'lon': lon[grid],
            'lat': lat[grid]
        }
        # 获取当前网格每个月的用水量，并添加到行数据中
        for month in range(num_months):
            row[f'{months_datas[month]}'] = water_value[month, grid]
        data.append(row)


    # 创建一个pandas DataFrame
    df = pd.DataFrame(data)
    # 将DataFrame保存为CSV文件
    df.to_csv(csv_file_path, index=False)
    print(df.columns)
    print(f"转换完成，CSV文件已保存为 {csv_file_path}")
    # 关闭 NetCDF 文件
    dataset.close()

def extract_YBHM_csv(csv_file_path, csv_YBHM_file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

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
        # 检查是否有至少四列
        if filtered_df.shape[1] < 4:
            raise ValueError("CSV文件的列数少于4列。")

        # 保留原有表头
        new_columns = list(filtered_df.columns[::])
        # 设置新的表头
        filtered_df.columns = new_columns

        # （可选）查看修改后的前几行数据
        print("修改后的表格前几行：")
        print(filtered_df.head())

        # 将筛选后的数据保存为新的CSV文件
        filtered_df.to_csv(csv_YBHM_file_path, index=False)
        print(f"\n筛选后的数据已保存到 '{csv_YBHM_file_path}'")

    except FileNotFoundError:
        print(f"文件未找到: {csv_file_path}")
    except pd.errors.EmptyDataError:
        print("CSV文件为空。")
    except KeyError as e:
        print(f"缺少必要的列: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def csv_to_points(csv_YBHM_file_path, points_YBHM_file_path):
    # 读取CSV文件
    csv_file = csv_YBHM_file_path  # 替换为你的CSV文件路径
    df = pd.read_csv(csv_file)

    # 检查CSV文件是否包含lon和lat列
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError("CSV文件必须包含'lon'和'lat'列")

    # 将lon和lat列转换为几何点
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # EPSG:4326表示WGS84坐标系

    # 保存为Shapefile
    gdf.to_file(points_YBHM_file_path, driver='ESRI Shapefile')

    print(f"Shapefile已保存到: {points_YBHM_file_path}")


if __name__ == "__main__":
    # ⭐withd
    # # withd irrigation
    # nc_file_path_0 = r'F:\021_1\1209296\irrigation water use v2\withd_irr_pcrglobwb.nc'
    # csv_file_path_1 = r'F:\022_1\withd\1_withd_irr.csv'
    # csv_env_file_path_2 = r'F:\022_1\withd\2_withd_irr_env.csv'
    # points_env_file_path_3 = r'F:\022_1\withd\3_withd_irr_env.shp'

    # withd livestock
    nc_file_path_0 = r'F:\021_1\1209296\livestock water use v2\withd_liv.nc'
    csv_file_path_1 = r'F:\022_1\withd\1_withd_liv.csv'
    csv_env_file_path_2 = r'F:\022_1\withd\2_withd_liv_env.csv'
    points_env_file_path_3 = r'F:\022_1\withd\3_withd_liv_env.shp'

    var_name = read_data(nc_file_path_0)
    print("var_name =", var_name[0])
    judge_file_path(csv_file_path_1)  # 判断结果文件的文件夹路径是否存在，并创建

    nc_to_csv(nc_file_path_0, csv_file_path_1, var_name[0])  # 1.nc转csv
    extract_YBHM_csv(csv_file_path_1, csv_env_file_path_2)  # 2.提取研究区区域CSV文件路径
    csv_to_points(csv_env_file_path_2, points_env_file_path_3)  # 3.将csv数据转为点数据
    # 4.计算印度的年用水量
