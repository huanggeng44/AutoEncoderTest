# 将已插值的城镇面积从csv转为tiff文件

import os
os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'

import pandas as pd
import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm

# 读取CSV文件(仅读取表头)
df = pd.read_csv("f:\\041_4\\GHS_B_S_csv\\urban_area_data_interpolated.csv", nrows=0)
print("CSV文件表头信息：")
print(list(df.columns))




def csv_to_multiband_tif(csv_path, output_tif):
    # 读取CSV文件
    print("正在读取CSV文件...")
    df = pd.read_csv(csv_path)

    # 提取经纬度列和数据列
    lons = df['lon'].values
    lats = df['lat'].values
    data_cols = [col for col in df.columns if col.startswith('E_')]

    # 计算栅格行列数
    unique_lons = np.unique(lons)
    unique_lats = np.unique(lats)
    cols = len(unique_lons)
    rows = len(unique_lats)

    # 创建多波段TIFF文件
    print("正在创建多波段TIFF文件...")
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, cols, rows, len(data_cols), gdal.GDT_Float32)

    # 设置地理参考信息
    xmin, xmax = min(unique_lons), max(unique_lons)
    ymin, ymax = min(unique_lats), max(unique_lats)
    xres = (xmax - xmin) / float(cols - 1)
    yres = (ymax - ymin) / float(rows - 1)

    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    out_ds.SetGeoTransform(geotransform)

    # 设置投影（WGS84）
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())

    # 将每列数据写入对应的波段
    print("正在写入波段数据...")
    for i, col in enumerate(tqdm(data_cols, desc="处理进度"), 1):
        # 创建二维数组
        band_data = np.zeros((rows, cols), dtype=np.float32)

        # 填充数据
        for idx, val in enumerate(df[col].values):
            x_idx = np.where(unique_lons == lons[idx])[0][0]
            y_idx = np.where(unique_lats == lats[idx])[0][0]
            band_data[y_idx, x_idx] = val

        # 写入波段
        band = out_ds.GetRasterBand(i)
        band.WriteArray(band_data)
        band.SetDescription(col)  # 设置波段描述为列名
        band.FlushCache()

    out_ds = None
    print(f"\n多波段TIFF文件已成功创建: {output_tif}")


if __name__ == "__main__":
    csv_path = "f:\\041_4\\GHS_B_S_csv\\urban_area_data_interpolated.csv"
    output_tif = "f:\\041_4\\GHS_B_S_csv\\urban_area_interpolated_1975_2020.tif"
    csv_to_multiband_tif(csv_path, output_tif)