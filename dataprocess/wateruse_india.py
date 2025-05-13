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

def extract_IND_csv(csv_file_path, csv_IND_file_path):
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
        filtered_df.to_csv(csv_IND_file_path, index=False)
        print(f"\n筛选后的数据已保存到 '{csv_IND_file_path}'")

    except FileNotFoundError:
        print(f"文件未找到: {csv_file_path}")
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

    # 将lon和lat列转换为几何点
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # EPSG:4326表示WGS84坐标系

    # 保存为Shapefile
    gdf.to_file(points_IND_file_path, driver='ESRI Shapefile')

    print(f"Shapefile已保存到: {points_IND_file_path}")

def IND_water_use_Year(IND_shp, water_use_shp, output_csv_file, output_points_file):
    """
    计算印度境内的逐年取用水量
    :param IND_shp: 印度国界
    :param water_use_shp: 月用水点数据
    :param output_file: 输出文件
    :return:
    """

    # 加载点数据（假设为WGS84坐标系，单位是度）
    points = gpd.read_file(water_use_shp)
    # 加载面数据（确保坐标系与点数据一致）
    polygon = gpd.read_file(IND_shp)
    # 对面几何体创建0.2度的缓冲区
    buffered_polygon = polygon.geometry.iloc[0].buffer(0.2)
    # #  空间筛选
    # mask = points.geometry.within(buffered_polygon)
    # selected_points = points[mask].copy()
    # #  保存掩膜结果
    # selected_points.to_file(
    #     output_points_file,
    #     driver="ESRI Shapefile",
    #     encoding='utf-8'
    # )

    # 将缓冲区转换为GeoDataFrame
    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_polygon], crs=polygon.crs)
    # 空间连接筛选位于缓冲区内的点
    points_in_buffer = gpd.sjoin(points, buffered_gdf, how="inner", predicate="within")
    # 提取所有字段名
    fields = points_in_buffer.columns.tolist()
    # 确定起始字段索引（第4个字段）
    start_idx = 3
    # 计算总年数（40年）
    total_years = 40

    # 初始化结果字典
    annual_totals = {}

    # 遍历每一年
    for y in range(total_years):
        # 计算当前年的字段范围
        field_start = start_idx + y * 12
        field_end = field_start + 12
        # 提取当前年的所有月度字段
        year_fields = fields[field_start:field_end]
        # 对每个点的年数据求和，再累加所有点
        annual_total = points_in_buffer[year_fields].sum().sum()
        # 存储结果（年份从1开始计数）
        year = year_fields[1][1:5]
        annual_totals[f"F_{year}"] = annual_total
        # annual_totals[f"F_{year}"] = annual_total/10000   # livestock取水数据10000    mining取水数据*10000倍
        # annual_totals[f"F_{year}"] = annual_total/1000    # manufacturing取水数据 * 1000

    print(annual_totals)

    df = pd.DataFrame(list(annual_totals.items()), columns=["Year", "Total_Water_Use"])
    df.to_csv(output_csv_file, index=False)
    print(df)


def IND_sector_water_use_year(file_groups , sectors_output):
    """
    计算各个部门的年取水量，并汇总到一个csv文件种
    :param sectors_output:
    :return:
    """

    def process_file(file_path):
        """
        处理单个长格式文件：
        输入文件格式：
        | year | water_total_use |
        |------|-----------------|
        | 1971 | 123.4           |
        ... (共40行)

        输出宽格式：
        | F1971 | F1972 | ... | F2010 |
        |-------|-------|-----|-------|
        | 123.4 | 456.7 | ... | 890.1 |
        """
        # 读取数据
        df = pd.read_csv(file_path)

        # 转换为宽格式
        df_wide = df.set_index('Year').T
        df_wide.columns = [f"{year}" for year in df_wide.columns]

        # 添加行名（文件名第9-11字符）
        filename = os.path.basename(file_path).split('.')[0]
        df_wide.index = [filename[8:11]]  # 索引从0开始计算

        return df_wide

    # 整合所有数据
    all_dfs  = []
    for group_name, files in file_groups.items():
        for file in files:
            df = process_file(file)
            all_dfs.append(df)
    combined_df = pd.concat(all_dfs)

    # 初始化部门汇总字典
    department_sums = {}

    # 计算每个部门的总和
    for department, files in file_groups.items():
        # 读取并合并该部门所有文件
        department_dfs = [process_file(file) for file in files]
        department_df = pd.concat(department_dfs)

        # 计算年度总和
        department_sum = department_df.sum().to_frame().T
        department_sum.index = [department]
        department_sums[department] = department_sum

    # 合并部门总和到总表
    final_df = pd.concat([combined_df,
                          department_sums["industry"],
                          department_sums["agriculture"],
                          department_sums["domestic"]])

    # 生成列顺序（F1971-F2010）
    # year_columns = [f"F{year}" for year in range(1971, 2011)]
    # final_df = final_df.reindex(columns=year_columns)

    # 输出到CSV
    final_df.to_csv(sectors_output)
    print(f"文件已保存：{sectors_output}")

if __name__ == "__main__":
    IND_shp = r'G:\DataSets\BasinRiverData\03_country\country\India.shp'
    # ⭐withd
    # # withd electriction    # elec取水数据照原
    # csv_file_path = r'F:\021_1\withd_elec\1_withd_elec.csv'  # 输出的CSV文件路径
    # csv_IND_file_path = r'F:\021_1\India\withd_elec\2_withd_elec_IND.csv'     # 提取研究区区域CSV文件路径
    # points_IND_file_path = r'F:\021_1\India\withd_elec\3_withd_elec_IND_P.shp'
    #
    # points_IND_year_file_path = r'F:\021_1\India\withd_elec\4_withd_elec_IND_Y.shp' # 用于计算年数据的印度内部的点（包括缓冲距离0.2°范围内的点）
    # csv_IND_year_file_path = r'F:\021_1\India\withd_elec\4_withd_elec_IND_Y.csv'
    #
    # extract_IND_csv(csv_file_path, csv_IND_file_path)
    # csv_to_points(csv_IND_file_path, points_IND_file_path)
    # IND_water_use_Year(IND_shp, points_IND_file_path, csv_IND_year_file_path, points_IND_year_file_path) # 计算印度逐年取水量

    # # withd domestic    # domestic取水数据照原
    # csv_file_path = r'F:\021_1\withd_dom\1_withd_dom.csv'  # 输出的CSV文件路径
    # csv_IND_file_path = r'F:\021_1\India\withd_dom\2_withd_dom_IND.csv'  # 提取研究区区域CSV文件路径
    # points_IND_file_path = r'F:\021_1\India\withd_dom\3_withd_dom_IND_P.shp'

    # points_IND_year_file_path = r'F:\021_1\India\withd_dom\4_withd_dom_IND_Y.shp'
    # csv_IND_year_file_path = r'F:\021_1\India\withd_dom\4_withd_dom_IND_Y.csv'

    # extract_IND_csv(csv_file_path, csv_IND_file_path)
    # csv_to_points(csv_IND_file_path, points_IND_file_path)
    # IND_water_use_Year(IND_shp, points_IND_file_path, csv_IND_year_file_path, points_IND_year_file_path) # 计算印度逐年取水量

    # # withd irrigation    # irrigation取水数据照原
    # csv_file_path = r'F:\021_1\withd_irr\1_withd_irr.csv'  # 输出的CSV文件路径
    # csv_IND_file_path = r'F:\021_1\India\withd_irr\2_withd_irr_IND.csv'  # 提取研究区区域CSV文件路径
    # points_IND_file_path = r'F:\021_1\India\withd_irr\3_withd_irr_IND_P.shp'

    # csv_IND_year_file_path = r'F:\021_1\India\withd_irr\4_withd_irr_IND_Y.csv'
    # points_IND_year_file_path = r'F:\021_1\India\withd_irr\4_withd_irr_IND_Y.shp'

    # extract_IND_csv(csv_file_path, csv_IND_file_path)
    # csv_to_points(csv_IND_file_path, points_IND_file_path)
    # IND_water_use_Year(IND_shp, points_IND_file_path, csv_IND_year_file_path, points_IND_year_file_path) # 计算印度逐年取水量

    # # # withd livestock    # livestock取水数据*10000倍
    # csv_file_path = r'F:\021_1\withd_liv\1_withd_liv.csv'  # 输出的CSV文件路径
    # csv_IND_file_path = r'F:\021_1\India\withd_liv\2_withd_liv_IND.csv'  # 提取研究区区域CSV文件路径
    # points_IND_file_path = r'F:\021_1\India\withd_liv\3_withd_liv_IND_P.shp'

    # csv_IND_year_file_path = r'F:\021_1\India\withd_liv\4_withd_liv_IND_Y.csv'
    # points_IND_year_file_path = r'F:\021_1\India\withd_liv\4_withd_liv_IND_Y.shp'

    # extract_IND_csv(csv_file_path, csv_IND_file_path)
    # csv_to_points(csv_IND_file_path, points_IND_file_path)
    # IND_water_use_Year(IND_shp, points_IND_file_path, csv_IND_year_file_path, points_IND_year_file_path) # 计算印度逐年取水量

    # # withd manufacturing    # manufacturing取水数据*1000
    # csv_file_path = r'F:\021_1\withd_mfg\1_withd_mfg.csv'  # 输出的CSV文件路径
    # csv_IND_file_path = r'F:\021_1\India\withd_mfg\2_withd_mfg_IND.csv'  # 提取研究区区域CSV文件路径
    # points_IND_file_path = r'F:\021_1\India\withd_mfg\3_withd_mfg_IND_P.shp'

    # csv_IND_year_file_path = r'F:\021_1\India\withd_mfg\4_withd_mfg_IND_Y.csv'
    # points_IND_year_file_path = r'F:\021_1\India\withd_mfg\4_withd_mfg_IND_Y.shp'

    # extract_IND_csv(csv_file_path, csv_IND_file_path)
    # csv_to_points(csv_IND_file_path, points_IND_file_path)
    # IND_water_use_Year(IND_shp, points_IND_file_path, csv_IND_year_file_path, points_IND_year_file_path) # 计算印度逐年取水量

    # withd mining   # mining取水数据*10000倍
    # csv_file_path = r'F:\021_1\withd_min\1_withd_min.csv'  # 输出的CSV文件路径
    # csv_IND_file_path = r'F:\021_1\India\withd_min\2_withd_min_IND.csv'  # 提取研究区区域CSV文件路径
    # points_IND_file_path = r'F:\021_1\India\withd_min\3_withd_min_IND_P.shp'
    #
    # csv_IND_year_file_path = r'F:\021_1\India\withd_min\4_withd_min_IND_Y.csv'
    # points_IND_year_file_path = r'F:\021_1\India\withd_min\4_withd_min_IND_Y.shp'

    # extract_IND_csv(csv_file_path, csv_IND_file_path)
    # csv_to_points(csv_IND_file_path, points_IND_file_path)
    # IND_water_use_Year(IND_shp, points_IND_file_path, csv_IND_year_file_path, points_IND_year_file_path)   # 计算印度逐年取水量
# =====================================================================================================================================
    csv_IND_year_file_path_min = r'F:\021_1\India\withd_min\4_withd_min_IND_Y.csv'
    csv_IND_year_file_path_mfg = r'F:\021_1\India\withd_mfg\4_withd_mfg_IND_Y.csv'
    csv_IND_year_file_path_ele = r'F:\021_1\India\withd_elec\4_withd_elec_IND_Y.csv'

    csv_IND_year_file_path_liv = r'F:\021_1\India\withd_liv\4_withd_liv_IND_Y.csv'
    csv_IND_year_file_path_irr = r'F:\021_1\India\withd_irr\4_withd_irr_IND_Y.csv'

    csv_IND_year_file_path_dom = r'F:\021_1\India\withd_dom\4_withd_dom_IND_Y.csv'

    file_groups = {
        "industry":[csv_IND_year_file_path_min, csv_IND_year_file_path_mfg, csv_IND_year_file_path_ele],  # 替换为实际路径
        "agriculture": [csv_IND_year_file_path_liv, csv_IND_year_file_path_irr],
        "domestic":  [csv_IND_year_file_path_dom]
    }

    sectors_output = r'F:\021_1\India\withd_sectors_IND.csv'

    IND_sector_water_use_year(file_groups, sectors_output)