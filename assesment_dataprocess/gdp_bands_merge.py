# 将GDP文件进行解压
# 裁剪GDPtiff文件
# 合并多波段数据

import os
os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'

import zipfile
import numpy as np
from osgeo import gdal, osr


def unzip_files(source_dir, target_dir):
    """解压所有ZIP文件到目标目录"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for item in os.listdir(source_dir):
        if item.endswith('.zip'):
            zip_path = os.path.join(source_dir, item)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"已解压: {item}")


def check_and_convert_projection(input_path, output_dir):
    """检查并转换坐标系为地理坐标系(WGS84)"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成输出路径
    output_path = os.path.join(output_dir, os.path.basename(input_path))

    ds = gdal.Open(input_path)
    src_proj = ds.GetProjection()
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_proj)

    # 如果是地理坐标系且是WGS84，直接返回原路径
    if src_srs.IsGeographic() and 'WGS 84' in src_srs.GetName():
        return input_path

    # 创建目标坐标系(WGS84)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)  # WGS84

    # 执行坐标转换并保存到指定目录
    gdal.Warp(output_path, ds, dstSRS=dst_srs)
    return output_path


def clip_tiff_files(source_dir, target_dir, bounds):
    """裁剪TIFF文件到指定范围"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 设置GDAL配置选项
    gdal.UseExceptions()

    # WGS84转换后的文件目录
    wgs84_dir = os.path.join(os.path.dirname(source_dir), "GDP_1992_2019_wgs84")

    for item in os.listdir(source_dir):
        if item.endswith('.tif'):
            input_path = os.path.join(source_dir, item)
            output_path = os.path.join(target_dir, item)

            # 检查并转换坐标系
            projected_path = check_and_convert_projection(input_path, wgs84_dir)

            # 使用GDAL进行裁剪
            options = gdal.TranslateOptions(
                projWin=[bounds['min_x'], bounds['max_y'], bounds['max_x'], bounds['min_y']]
            )
            gdal.Translate(output_path, projected_path, options=options)
            print(f"已裁剪: {item}")


def merge_tiff_files(source_dir, output_path):
    """将多个单波段TIFF文件合并为多波段栅格"""
    # 获取所有TIFF文件
    tif_files = [f for f in os.listdir(source_dir) if f.endswith('.tif')]
    if not tif_files:
        print("未找到TIFF文件")
        return

    # 读取第一个文件作为模板
    first_file = os.path.join(source_dir, tif_files[0])
    ds = gdal.Open(first_file)
    driver = ds.GetDriver()
    rows, cols = ds.RasterYSize, ds.RasterXSize

    # 创建多波段输出文件
    out_ds = driver.Create(output_path, cols, rows, len(tif_files), gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # 逐个添加波段
    for i, tif_file in enumerate(tif_files, 1):
        file_path = os.path.join(source_dir, tif_file)
        band_ds = gdal.Open(file_path)
        band_data = band_ds.GetRasterBand(1).ReadAsArray()

        # 获取原始nodata值
        nodata = band_ds.GetRasterBand(1).GetNoDataValue()

        out_band = out_ds.GetRasterBand(i)
        out_band.WriteArray(band_data)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)  # 设置nodata值
        out_band.SetDescription(os.path.splitext(tif_file)[0])
        out_band.FlushCache()

    out_ds = None
    print(f"已合并 {len(tif_files)} 个波段到 {output_path}")


if __name__ == "__main__":
    # 配置路径和参数
    source_dir = r"f:\03全球GDP数据_1\Real GDP\updated real GDP"
    gdp_dir = os.path.join(source_dir, "GDP_1992_2019")
    clipped_dir = os.path.join(source_dir, "GDP_1992_2019_YBHM")

    # 裁剪范围 (72.3°E, 21°N, 98.5°E, 32.6°N)
    clip_bounds = {
        'min_x': 72.3,  # 最小经度
        'max_x': 98.5,  # 最大经度
        'min_y': 21.0,  # 最小纬度
        'max_y': 32.6  # 最大纬度
    }

    # 执行解压和裁剪
    # unzip_files(source_dir, gdp_dir)
    # clip_tiff_files(gdp_dir, clipped_dir, clip_bounds)

    # 新增合并操作
    merged_output = os.path.join(clipped_dir, "gdp_1992_2019_gdp.tif")
    merge_tiff_files(clipped_dir, merged_output)

    print("处理完成！")