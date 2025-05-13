# 由于城镇面积每5年一个数据，因此进行中间年份的插值计算
# 首先将tif文件转为csv文件

import os
os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'

import numpy as np
import pandas as pd
from osgeo import gdal


def resample_tif(input_file, output_file, target_resolution=0.01):
    """使用双线性插值重采样TIFF文件"""
    options = gdal.WarpOptions(
        xRes=target_resolution,
        yRes=target_resolution,
        resampleAlg='bilinear',  # 明确指定双线性插值
        format='GTiff'
    )
    gdal.Warp(output_file, input_file, options=options)


def tif_to_csv(input_dir, output_dir, resampled_dir):
    # 确保输出目录存在
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(resampled_dir, exist_ok=True)

    # 获取所有tif文件并按文件名排序
    tif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])

    # if not tif_files:
    #     print("未找到任何TIFF文件")
    #     return

    # # 重采样所有文件到指定目录
    # for tif_file in tif_files:
    #     input_path = os.path.join(input_dir, tif_file)
    #     output_path = os.path.join(resampled_dir, tif_file)
    #     resample_tif(input_path, output_path)

    # 读取第一个重采样后的文件获取地理信息
    first_ds = gdal.Open(os.path.join(resampled_dir, tif_files[0]))
    transform = first_ds.GetGeoTransform()
    cols = first_ds.RasterXSize
    rows = first_ds.RasterYSize

    # 创建坐标网格
    x = np.linspace(transform[0], transform[0] + (cols - 1) * transform[1], cols)
    y = np.linspace(transform[3], transform[3] + (rows - 1) * transform[5], rows)
    xx, yy = np.meshgrid(x, y)

    # 初始化DataFrame
    df = pd.DataFrame({
        'lon': xx.flatten(),
        'lat': yy.flatten()
    })

    # 处理每个重采样后的TIFF文件
    for tif_file in tif_files:
        print(tif_file)
        # 从文件名提取年份
        year = ''.join(filter(str.isdigit, tif_file))[:4]

        # 读取重采样后的数据
        ds = gdal.Open(os.path.join(resampled_dir, tif_file))
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        nodata = band.GetNoDataValue()

        # 将数据添加到DataFrame，列名格式为E_年份
        df[f"E_{year}"] = data.flatten()
        print(year)

        # 处理nodata值
        if nodata is not None:
            df.loc[df[f"E_{year}"] == nodata, f"E_{year}"] = np.nan

    # 保存为CSV
    output_file = os.path.join(output_dir, "urban_area_data.csv")
    df.to_csv(output_file, index=False)
    print(f"已成功转换并保存到 {output_file}")


if __name__ == "__main__":
    input_dir = "f:\\041_4\\GHB_B_S_1975_2030_0083"  # 输入目录
    output_dir = "f:\\041_4\\GHS_B_S_csv"  # CSV输出目录
    resampled_dir = "f:\\041_4\\GHS_B_S_1975_2030_01"  # 重采样文件输出目录
    tif_to_csv(input_dir, output_dir, resampled_dir)