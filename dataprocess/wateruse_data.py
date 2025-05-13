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

def elec_nc_to_csv(nc_file_path, csv_file_path):
    # NC文件转csv文件
    # # 打开NetCDF文件
    dataset = nc.Dataset(nc_file_path)
    # # 读取变量
    lon = dataset.variables['lon'][:]  # 经度，形状为 (grid_num,)
    lat = dataset.variables['lat'][:]  # 纬度，形状为 (grid_num,)
    cons_elec = dataset.variables['cons_elec']  # 用水量，形状为 (month, grid_num)
    months = dataset.variables['month'][:]  # 月份信息，形状为 (month,)，如果月份是索引，可以根据需要处理

    # 获取 'months since 1971-1' 的具体信息
    months_datas = read_months_info(nc_file_path)

    # 获取维度大小
    num_grids = len(lon)
    num_months = cons_elec.shape[0]

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
            row[f'{months_datas[month]} '] = cons_elec[month, grid]
        print(row)
        data.append(row)

    # 创建一个pandas DataFrame
    df = pd.DataFrame(data)
    # 将DataFrame保存为CSV文件
    df.to_csv(csv_file_path, index=False)

    print(f"转换完成，CSV文件已保存为 {csv_file_path}")

    # 关闭 NetCDF 文件
    dataset.close()


    # # 检查输出CSV文件
    # # 读取CSV文件
    # df = pd.read_csv(csv_file_path)
    # # 显示整个表格
    # print(df)

def elec_extract_YBHM_csv(csv_file_path, csv_YBHM_file_path, new_datas_headers):
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
        filtered_df = df[(df['lon'] >= 72) & (df['lon'] <= 99) &
                         (df['lat'] >= 21) & (df['lat'] <= 33)]
        # 检查是否有至少四列
        if filtered_df.shape[1] < 4:
            raise ValueError("CSV文件的列数少于4列。")
        # 选择从第四列开始的列，并将其数值保留四位小数
        filtered_df.iloc[:, 3:] = filtered_df.iloc[:, 3:].applymap(lambda x: round(x * 1000, 4) if pd.notnull(x) else x)
        # filtered_df.iloc[:, 3:] = filtered_df.iloc[:, 3:].round(4)

        # 显示筛选后的表格基本信息（可选）
        # print("\n=== 筛选后表格基本信息 ===")
        # print(filtered_df.info())
        # print("\n=== 筛选后表格前几行 ===")
        # print(filtered_df.head())

        # 检查新表头的长度是否与需要替换的列数匹配
        if len(new_datas_headers) < (filtered_df.shape[1] - 3):
            raise ValueError("新表头的长度不足以替换从第四列开始的列。")

        # 保留前三列的原有表头
        new_columns = list(filtered_df.columns[:3]) + new_datas_headers

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


def irr_extract_YBHM_csv(csv_file_path, csv_YBHM_file_path, new_datas_headers):
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
        filtered_df = df[(df['lon'] >= 72) & (df['lon'] <= 99) &
                         (df['lat'] >= 21) & (df['lat'] <= 33)]
        # 检查是否有至少四列
        if filtered_df.shape[1] < 4:
            raise ValueError("CSV文件的列数少于4列。")
        # 选择从第四列开始的列，并将其数值保留四位小数
        filtered_df.iloc[:, 3:] = filtered_df.iloc[:, 3:].round(4)

        # 显示筛选后的表格基本信息（可选）
        # print("\n=== 筛选后表格基本信息 ===")
        # print(filtered_df.info())
        # print("\n=== 筛选后表格前几行 ===")
        # print(filtered_df.head())

        # 检查新表头的长度是否与需要替换的列数匹配
        if len(new_datas_headers) < (filtered_df.shape[1] - 3):
            raise ValueError("新表头的长度不足以替换从第四列开始的列。")

        # 保留前三列的原有表头
        new_columns = list(filtered_df.columns[:3]) + new_datas_headers

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


# ==============================================================分割线=========================================================================
def nc_to_csv(nc_file_path, csv_file_path,variable_name):
    # NC文件转csv文件
    # # 打开NetCDF文件
    dataset = nc.Dataset(nc_file_path)
    # # 读取变量
    lon = dataset.variables['lon'][:]  # 经度，形状为 (grid_num,)
    lat = dataset.variables['lat'][:]  # 纬度，形状为 (grid_num,)
    cons_elec = dataset.variables[variable_name]  # 用水量，形状为 (month, grid_num)
    months = dataset.variables['month'][:]  # 月份信息，形状为 (month,)，如果月份是索引，可以根据需要处理

    # 获取 'months since 1971-1' 的具体信息
    months_datas = read_months_info(nc_file_path)  # 重新设置月用水量字段的字段名

    # 获取维度大小
    num_grids = len(lon)
    num_months = cons_elec.shape[0]

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
            # row[f'{months_datas[month]} '] = (cons_elec[month, grid]*100000).round(4)     # domestic的用水值*100,000倍
            # row[f'{months_datas[month]} '] = (cons_elec[month, grid]*100).round(4)        # livestock的用水值*100倍
            # row[f'{months_datas[month]} '] = (cons_elec[month, grid] * 1000).round(4)       # manufacturing的用水值*1000倍   # manufacturing取水数据*1000
            row[f'{months_datas[month]} '] = (cons_elec[month, grid] * 10000).round(4)    # mining的用水值*10000倍         # livestock取水值*10000倍     # mining取水值*10000倍
            # row[f'{months_datas[month]} '] = cons_elec[month, grid]
            # print(cons_elec[month, grid])
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
        filtered_df = df[(df['lon'] >= 72) & (df['lon'] <= 99) &
                         (df['lat'] >= 21) & (df['lat'] <= 33)]
        # 检查是否有至少四列
        if filtered_df.shape[1] < 4:
            raise ValueError("CSV文件的列数少于4列。")
        # 选择从第四列开始的列，并将其数值保留四位小数
        filtered_df.iloc[:, 3:] = filtered_df.iloc[:, 3:].round(4)

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


def spatial_connection(subbasin_shape_file_path, points_YBHM_file_path, spatial_output, spatial_csv_output):
    subbasin_pd = gpd.read_file(subbasin_shape_file_path)
    cons_water_df = gpd.read_file(points_YBHM_file_path)

    # 进行空间连接
    joined = gpd.sjoin(subbasin_pd, cons_water_df, how='left')
    print(joined.columns)

    # 对取用水量字段进行求总和计算
    columns_to_sum = joined.columns[7:]  # 第8列是从索引7开始的
    joined[columns_to_sum] = joined.groupby('HYBAS_ID')[columns_to_sum].transform('sum')

    # 删除重复的面矢量数据行
    unique_row = joined.drop_duplicates(subset=['HYBAS_ID'])
    print(unique_row)

    # 检查是否有至少四列
    if unique_row.shape[1] < 8:
        raise ValueError("CSV文件的列数少于4列。")

    # 选择从第四列开始的列，并将其数值保留两位小数
    unique_row.iloc[:, 7:] = unique_row.iloc[:, 7:].round(2)

    # 将 DataFrame 转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(unique_row, geometry='geometry', crs="EPSG:4326")
    # 将 GeoDataFrame 写入 Shapefile
    gdf.to_file(spatial_output)
    print(f"Shapefile 已成功保存到 {spatial_output}")

    df = gdf.drop(columns='geometry')
    df.to_csv(spatial_csv_output, index=False)
    print(f"CSV 已成功保存到 {spatial_csv_output}")


def culculate_year_cons_water(input_polygon_file, output_polygon_file):
    # 1. 读取面数据
    gdf = gpd.read_file(input_polygon_file)  # 替换为你的面数据文件路径
    # 打印所有字段名（可选，用于确认字段顺序）
    print("所有字段名：")
    print(gdf.columns.tolist())

    # 保留其余字段的第一个值
    other_cols = gdf.columns[:6]  # 前6个字段
    geo_cols = gdf.columns[-1]
    remaining_cols = gdf.columns[6:-1]

    # 创建一个新的GeoDataFrame
    new_gdf = gdf[other_cols].copy()  # 创建一个新的GeoDataFrame，保留前6个字段
    new_gdf['geometry'] = gdf[geo_cols]

    # 将480个字段每12个字段分为一组，共40组
    grouped_columns = [remaining_cols[i:i + 12] for i in range(0, len(remaining_cols), 12)]

    # 对每组字段进行求和，并将结果存储到新列中
    for i, group in enumerate(grouped_columns):
        # 新字段名为原组内第一个字段名的前5个字符
        new_column_name = group[0][:5]
        new_gdf[new_column_name] = gdf[group].sum(axis=1)

    print(new_gdf.columns)

    # 转换为GeoDataFrame
    geo_df = gpd.GeoDataFrame(new_gdf, geometry='geometry', crs="EPSG:4326")
    # 保存结果到新的Shapefile文件
    geo_df.to_file(output_polygon_file)

    print(f"处理完成，结果已保存到: {output_polygon_file}")

def three_sectors_cons_process(input_files_list, output_shapefile):
    # 读取所有shapefile文件
    gdfs = [gpd.read_file(path) for path in input_files_list]
    # 定义每个文件的除数
    if len(input_files_list) == 2:
        divisors = [1, 100]     # (irr, liv)=(1, 100)
    elif len(input_files_list) == 3:
        divisors = [1000, 1000, 10000] # (elec, mfg, min)=(1000, 1000, 10000)
    elif len(input_files_list) == 1:
        divisors = [100000]     # dom = 100000

    processed_gdfs = []
    geometry_col = ""
    geometries = []


    for idx, (gdf, divisor) in enumerate(zip(gdfs, divisors), start=1):
        if 'HYBAS_ID' not in gdf.columns:
            print(f"文件 {input_files_list[idx - 1]} 缺少 'HYBAS_ID' 字段，跳过该文件。")
            continue

        # 选择前三个字段
        prefix_fields = gdf.columns[:2].tolist()
        # 选择需要处理的字段（第7个字段及之后）
        fields_to_process = gdf.columns[6:-1].tolist()
        # 最后一个字段
        geometry_col = gdf.columns[-1]
        # 分离几何列
        geometries = gdf[geometry_col]


        # 进行除法操作
        gdf_processed = gdf[prefix_fields + fields_to_process].copy()
        gdf_processed[fields_to_process] = gdf_processed[fields_to_process].apply(pd.to_numeric, errors='coerce')
        gdf_processed[fields_to_process] = (gdf_processed[fields_to_process] / divisor).round(2)

        processed_gdfs.append(gdf_processed)

    if not processed_gdfs:
        print("没有有效的shapefile文件可用于处理。")
        return

    # 初始化结果为第一个处理后的DataFrame
    result_gdf = processed_gdfs[0].copy()

    # 依次合并并相加后续的DataFrame
    for gdf in processed_gdfs[1:]:
        # 合并时基于HYBAS_ID进行左连接
        result_gdf = pd.merge(
            result_gdf,
            gdf,
            on='HYBAS_ID',
            how='left',
            suffixes=('_1', '_2')   # 避免字段名冲突
        )

        # 保留前三个字段和所有处理后的字段
        second_column = gdf.columns[1]
        result_gdf[second_column] = result_gdf[f'{second_column}_1']
        result_gdf.drop(columns=[f'{second_column}_1', f'{second_column}_2'], inplace=True)

        # 处理字段相加，避免重复列名
        fields_to_add = gdf.columns[2:]  # 排除前两个字段(HYBAS_ID, Basin_Name)
        for field in fields_to_add:
            col1 = f'{field}_1'
            col2 = f'{field}_2'
            if col2 in result_gdf:
                result_gdf[field] = (result_gdf[col1].fillna(0) + result_gdf[col2].fillna(0)).round(2)
                result_gdf.drop(columns=[col1, col2], inplace=True)
            else:
                result_gdf[field] = result_gdf[col1].fillna(0).round(2)


    # 合并几何列回去
    result_gdf[geometry_col] = geometries
    # 转换为 GeoDataFrame
    shp_gdf = gpd.GeoDataFrame(result_gdf, geometry=geometries)
    # 保存结果到新的shapefile
    shp_gdf.to_file(output_shapefile)

    print(f"处理完成，结果已保存到 '{output_shapefile}'")


def total_cons_process(input_files_list, output_shapefile):
    # 读取所有shapefile文件
    gdfs = [gpd.read_file(path) for path in input_files_list]

    processed_gdfs = []
    geometry_col = ""
    geometries = []

    for idx, gdf in enumerate(gdfs, start=1):
        if 'HYBAS_ID' not in gdf.columns:
            print(f"文件 {input_files_list[idx - 1]} 缺少 'HYBAS_ID' 字段，跳过该文件。")
            continue

        # 选择前三个字段
        prefix_fields = gdf.columns[:2].tolist()
        # 选择需要处理的字段（第7个字段及之后）
        fields_to_process = gdf.columns[2:-1].tolist()
        # 最后一个字段
        geometry_col = gdf.columns[-1]
        # 分离几何列
        geometries = gdf[geometry_col]

        gdf_processed = gdf[prefix_fields + fields_to_process].copy()
        processed_gdfs.append(gdf_processed)

    if not processed_gdfs:
        print("没有有效的shapefile文件可用于处理。")
        return

    # 初始化结果为第一个处理后的DataFrame
    result_gdf = processed_gdfs[0].copy()

    # 依次合并并相加后续的DataFrame
    for gdf in processed_gdfs[1:]:
        # 合并时基于HYBAS_ID进行左连接
        result_gdf = pd.merge(
            result_gdf,
            gdf,
            on='HYBAS_ID',
            how='left',
            suffixes=('_1', '_2')  # 避免字段名冲突
        )

        # 保留前三个字段和所有处理后的字段
        second_column = gdf.columns[1]
        result_gdf[second_column] = result_gdf[f'{second_column}_1']
        result_gdf.drop(columns=[f'{second_column}_1', f'{second_column}_2'], inplace=True)

        # 处理字段相加，避免重复列名
        fields_to_add = gdf.columns[2:]  # 排除前两个字段(HYBAS_ID, Basin_Name)
        for field in fields_to_add:
            col1 = f'{field}_1'
            col2 = f'{field}_2'
            if col2 in result_gdf:
                result_gdf[field] = (result_gdf[col1].fillna(0) + result_gdf[col2].fillna(0)).round(2)
                result_gdf.drop(columns=[col1, col2], inplace=True)
            else:
                result_gdf[field] = result_gdf[col1].fillna(0).round(2)

    # 合并几何列回去
    result_gdf[geometry_col] = geometries
    # 转换为 GeoDataFrame
    shp_gdf = gpd.GeoDataFrame(result_gdf, geometry=geometries)
    # 保存结果到新的shapefile
    shp_gdf.to_file(output_shapefile)

    print(f"处理完成，结果已保存到 '{output_shapefile}'")

def shp_to_csv(shp_list, csv_output):
    # 读取四个Shapefile文件
    shp1 = gpd.read_file(shp_list[0]).drop(columns=['geometry'])
    shp2 = gpd.read_file(shp_list[1]).drop(columns=['geometry'])
    shp3 = gpd.read_file(shp_list[2]).drop(columns=['geometry'])
    shp4 = gpd.read_file(shp_list[3]).drop(columns=['geometry'])

    # 重命名列名
    def rename_columns(df, prefix):
        columns = df.columns.tolist()
        new_columns = [columns[0], columns[1]] + [f"{prefix}{n}" for n in range(1971, 2011)]
        df.columns = new_columns
        return df

    shp1 = rename_columns(shp1, 'T')
    shp2 = rename_columns(shp2, 'A')
    shp3 = rename_columns(shp3, 'I')
    shp4 = rename_columns(shp4, 'D')

    # 对第三列及以后的数值保留两位小数
    shp1.iloc[:, 2:] = shp1.iloc[:, 2:].round(2)
    shp2.iloc[:, 2:] = shp2.iloc[:, 2:].round(2)
    shp3.iloc[:, 2:] = shp3.iloc[:, 2:].round(2)
    shp4.iloc[:, 2:] = shp4.iloc[:, 2:].round(2)

    # 仅保留第一个文件的basin_ID和basin_name列
    shp1 = shp1[['HYBAS_ID', 'Basin_Name'] + [f"T{n}" for n in range(1971, 2011)]]
    shp2 = shp2[[f"A{n}" for n in range(1971, 2011)]]
    shp3 = shp3[[f"I{n}" for n in range(1971, 2011)]]
    shp4 = shp4[[f"D{n}" for n in range(1971, 2011)]]

    # 合并四个文件
    merged_df = pd.concat([shp1, shp2, shp3, shp4], axis=1)

    # 保存为CSV文件
    merged_df.to_csv(csv_output, index=False)

    print("CSV文件已生成！")

def add_time_sas(csv_output_file_path, add_time_sas_file_path):
    """
    为创建好的各部门用水情况表格增加时间列，以便在sas软件中进行分析处理
    :param csv_output_file_path:
    :param add_time_sas_file_path:
    :return:
    """
    # 读取原始CSV文件
    df = pd.read_csv(csv_output_file_path)

    # 添加新列
    for year in range(1971, 2011):
        column_name = f'Y{year}'
        df[column_name] = year  # 整列填充年份数值

    # 保存修改后的CSV文件（不保留索引）
    df.to_csv(add_time_sas_file_path, index=False)

if __name__ == "__main__":
    # subbasin_shape_file_path = r"G:\BasicDatasets\YBHM_sub_catchments.shp"

    # ⭐cons
    # # electriction    # electriction用水数据*1000倍
    # nc_file_path = r'F:\021_1\1209296\electricity water use v2\cons_elec.nc'    # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\cons_elec\1_cons_elec.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\cons_elec\2_cons_elec_YBHM.csv'     # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\cons_elec\3_cons_elec_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\cons_elec\4_cons_elec_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\cons_elec\4_cons_elec_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\cons_elec\5_cons_elec_YBHM_year.shp'
    # # electriction处理方法
    # read_data(nc_file_path)   # 浏览数据
    # elec_nc_to_csv(nc_file_path, csv_file_path)    # nc转csv（elec的nc转tif尚未进行倍数和保留小数的处理，同时月份的用水值字段名未更改至目标格式）
    # months_datas = read_months_info(nc_file_path) # 重新设置月用水量字段的字段名
    # elec_extract_YBHM_csv(csv_file_path, csv_YBHM_file_path, months_datas) # 提取研究区数据（elec提取研究区数据做了扩大了1000倍扩大处理以及保留4位小数的处理，同时修改用水值字段名）
    # csv_to_points(csv_YBHM_file_path, points_YBHM_file_path)  # 将csv数据转为点数据
    # spatial_connection(subbasin_shape_file_path, points_YBHM_file_path, spatial_cons_water_file_path,
    #                    spatial_cons_water_csv_file_path)  # 空间连接
    # culculate_year_cons_water(spatial_cons_water_file_path, year_cons_water_file_path)


    # irrigation    # irrigation用水数据照原
    # nc_file_path = r'F:\021_1\1209296\irrigation water use v2\cons_irr_h08.nc'
    # csv_file_path = r'F:\021_1\cons_irr\1_cons_irr.csv'
    # csv_YBHM_file_path = r'F:\021_1\cons_irr\2_cons_irr_YBHM.csv'
    # points_YBHM_file_path = r'F:\021_1\cons_irr\3_cons_irr_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\cons_irr\4_cons_irr_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\cons_irr\4_cons_irr_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\cons_irr\5_cons_irr_YBHM_year.shp'
    # irrigation处理方法
    # var_name = read_data(nc_file_path)   # 浏览数据
    # irr_nc_to_csv(nc_file_path, csv_file_path, var_name[0])  # nc转csv
    # months_datas = read_months_info(nc_file_path) # 重新设置月用水量字段的字段名
    # irr_extract_YBHM_csv(csv_file_path, csv_YBHM_file_path, months_datas) # 提取研究区数据（irr提取研究区数据做了保留4位小数的处理，同时修改用水值字段名）
    # csv_to_points(csv_YBHM_file_path, points_YBHM_file_path)  # 将csv数据转为点数据
    # spatial_connection(subbasin_shape_file_path, points_YBHM_file_path, spatial_cons_water_file_path,
    #                    spatial_cons_water_csv_file_path)  # 空间连接
    # culculate_year_cons_water(spatial_cons_water_file_path, year_cons_water_file_path)

    # # domestic  # domestic的用水值*100000倍
    # nc_file_path = r'F:\021_1\1209296\domestic water use v2\cons_dom.nc'
    # csv_file_path = r'F:\021_1\cons_dom\1_cons_dom.csv'
    # csv_YBHM_file_path = r'F:\021_1\cons_dom\2_cons_dom_YBHM.csv'
    # points_YBHM_file_path = r'F:\021_1\cons_dom\3_cons_dom_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\cons_dom\4_cons_dom_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\cons_dom\4_cons_dom_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\cons_dom\5_cons_dom_YBHM_year.shp'

    # # livestock  # livestock的用水值*100倍
    # nc_file_path = r'F:\021_1\1209296\livestock water use v2\cons_liv.nc'
    # csv_file_path = r'F:\021_1\cons_liv\1_cons_liv.csv'
    # csv_YBHM_file_path = r'F:\021_1\cons_liv\2_cons_liv_YBHM.csv'
    # points_YBHM_file_path = r'F:\021_1\cons_liv\3_cons_liv_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\cons_liv\4_cons_liv_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\cons_liv\4_cons_liv_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\cons_liv\5_cons_liv_YBHM_year.shp'

    # # manufacturing  # manufacturing的用水值*1000
    # nc_file_path = r'F:\021_1\1209296\manufacturing water use v2\cons_mfg.nc'
    # csv_file_path = r'F:\021_1\cons_mfg\1_cons_mfg.csv'
    # csv_YBHM_file_path = r'F:\021_1\cons_mfg\2_cons_mfg_YBHM.csv'
    # points_YBHM_file_path = r'F:\021_1\cons_mfg\3_cons_mfg_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\cons_mfg\4_cons_mfg_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\cons_mfg\4_cons_mfg_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\cons_mfg\5_cons_mfg_YBHM_year.shp'

    # # mining  # mining的用水值*10000
    # nc_file_path = r'F:\021_1\1209296\mining water use v2\cons_min.nc'
    # csv_file_path = r'F:\021_1\cons_min\1_cons_min.csv'
    # csv_YBHM_file_path = r'F:\021_1\cons_min\2_cons_min_YBHM.csv'
    # points_YBHM_file_path = r'F:\021_1\cons_min\3_cons_min_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\cons_min\4_cons_min_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\cons_min\4_cons_min_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\cons_min\5_cons_min_YBHM_year.shp'

    # ⭐withd
    # # withd electriction    # elec取水数据照原
    # nc_file_path = r'F:\021_1\1209296\electricity water use v2\withd_elec.nc'    # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\withd_elec\1_withd_elec.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\withd_elec\2_withd_elec_YBHM.csv'     # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\withd_elec\3_withd_elec_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\withd_elec\4_withd_elec_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\withd_elec\4_withd_elec_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\withd_elec\5_withd_elec_YBHM_year.shp'

    # # withd domestic    # domestic取水数据照原
    # nc_file_path = r'F:\021_1\1209296\domestic water use v2\withd_dom.nc'  # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\withd_dom\1_withd_dom.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\withd_dom\2_withd_dom_YBHM.csv'  # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\withd_dom\3_withd_dom_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\withd_dom\4_withd_dom_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\withd_dom\4_withd_dom_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\withd_dom\5_withd_dom_YBHM_year.shp'

    # # withd irrigation    # irrigation取水数据照原
    # nc_file_path = r'F:\021_1\1209296\irrigation water use v2\withd_irr_h08.nc'  # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\withd_irr\1_withd_irr.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\withd_irr\2_withd_irr_YBHM.csv'  # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\withd_irr\3_withd_irr_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\withd_irr\4_withd_irr_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\withd_irr\4_withd_irr_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\withd_irr\5_withd_irr_YBHM_year.shp'

    # # # withd livestock    # livestock取水数据*10000倍
    # nc_file_path = r'F:\021_1\1209296\livestock water use v2\withd_liv.nc'  # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\withd_liv\1_withd_liv.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\withd_liv\2_withd_liv_YBHM.csv'  # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\withd_liv\3_withd_liv_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\withd_liv\4_withd_liv_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\withd_liv\4_withd_liv_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\withd_liv\5_withd_liv_YBHM_year.shp'

    # # # withd manufacturing    # manufacturing取水数据*1000
    # nc_file_path = r'F:\021_1\1209296\manufacturing water use v2\withd_mfg.nc'  # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\withd_mfg\1_withd_mfg.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\withd_mfg\2_withd_mfg_YBHM.csv'  # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\withd_mfg\3_withd_mfg_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\withd_mfg\4_withd_mfg_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\withd_mfg\4_withd_mfg_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\withd_mfg\5_withd_mfg_YBHM_year.shp'

    # # # withd mining   # mining取水数据*10000倍
    # nc_file_path = r'F:\021_1\1209296\mining water use v2\withd_min.nc'  # 原始数据NC文件路径
    # csv_file_path = r'F:\021_1\withd_min\1_withd_min.csv'  # 输出的CSV文件路径
    # csv_YBHM_file_path = r'F:\021_1\withd_min\2_withd_min_YBHM.csv'  # 提取研究区区域CSV文件路径
    # points_YBHM_file_path = r'F:\021_1\withd_min\3_withd_min_YBHM_P.shp'
    # spatial_cons_water_file_path = r'F:\021_1\withd_min\4_withd_min_YBHM_sptlcn.shp'
    # spatial_cons_water_csv_file_path = r'F:\021_1\withd_min\4_withd_min_YBHM_sptlcn_csv.csv'
    # year_cons_water_file_path = r'F:\021_1\withd_min\5_withd_min_YBHM_year.shp'

    # # ⭐读取nc数据，并且将月数据处理为年数据
    # var_name = read_data(nc_file_path)   # 浏览数据
    # print("var_name =", var_name[0])
    # judge_file_path(csv_file_path)  # 判断结果文件的文件夹路径是否存在，并创建
    # nc_to_csv(nc_file_path, csv_file_path, var_name[0])       # 1.nc转csv
    # extract_YBHM_csv(csv_file_path, csv_YBHM_file_path)       # 2.提取研究区区域CSV文件路径
    # csv_to_points(csv_YBHM_file_path, points_YBHM_file_path)  # 3.将csv数据转为点数据
    # spatial_connection(subbasin_shape_file_path, points_YBHM_file_path, spatial_cons_water_file_path, spatial_cons_water_csv_file_path)  # 空间连接
    # culculate_year_cons_water(spatial_cons_water_file_path, year_cons_water_file_path)

    # ⭐将6个部门的用水数据，通过相加处理为农业用水、工业用水、生活用水三大类
    three_sectors_cons_folder = r"F:\021_1\three_sectors_cons"
    # # 农业用水 = irr + liv
    # year_cons_irr_water_file_path = r'F:\021_1\cons_irr\5_cons_irr_YBHM_year.shp'
    # year_cons_liv_water_file_path = r'F:\021_1\cons_liv\5_cons_liv_YBHM_year.shp'
    # aggriculture_cons_files_list = [year_cons_irr_water_file_path, year_cons_liv_water_file_path]
    aggriculture_cons_file_path = os.path.join(three_sectors_cons_folder, "aggriculture_cons.shp")
    # three_sectors_cons_process(aggriculture_cons_files_list, aggriculture_cons_file_path)
    # # 工业用水 = elec + mfg + min
    # year_cons_elec_water_file_path = r'F:\021_1\cons_elec\5_cons_elec_YBHM_year.shp'
    # year_cons_mfg_water_file_path = r'F:\021_1\cons_mfg\5_cons_mfg_YBHM_year.shp'
    # year_cons_min_water_file_path = r'F:\021_1\cons_min\5_cons_min_YBHM_year.shp'
    # industry_cons_files_list = [year_cons_elec_water_file_path, year_cons_mfg_water_file_path, year_cons_min_water_file_path]
    industry_cons_file_path = os.path.join(three_sectors_cons_folder, "industry_cons.shp")
    # three_sectors_cons_process(industry_cons_files_list, industry_cons_file_path)
    # # 生活用水 = dom
    # year_cons_dom_water_file_path = r'F:\021_1\cons_dom\5_cons_dom_YBHM_year.shp'
    # domestic_cons_files_list = [year_cons_dom_water_file_path]
    domestic_cons_file_path = os.path.join(three_sectors_cons_folder, "domestic_cons.shp")
    # three_sectors_cons_process(domestic_cons_files_list, domestic_cons_file_path)
    # 总用水量total = aggriculture + industry + domestic
    sectors_cons_files_list = [aggriculture_cons_file_path, industry_cons_file_path, domestic_cons_file_path]
    total_cons_file_path = os.path.join(three_sectors_cons_folder, "three_sectors_total_cons.shp")
    # total_cons_process(sectors_cons_files_list, total_cons_file_path)


    # ⭐将三个部门用水数据及总用水数据的shp文件输出为一个csv文件
    shp_list = [total_cons_file_path, aggriculture_cons_file_path, industry_cons_file_path, domestic_cons_file_path]
    csv_output_file_path = os.path.join(three_sectors_cons_folder, "three_sectors_cons.csv")
    # shp_to_csv(shp_list, csv_output_file_path)
    add_time_sas_file_path  = os.path.join(three_sectors_cons_folder, "three_sectors_cons_sas.csv")
    add_time_sas(csv_output_file_path, add_time_sas_file_path)













