# 本文件用于将【多个单波段人口栅格】合并【为一个多波段人口栅格】,并且裁剪除研究区env区域
import os
os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'

from osgeo import gdal, osr

# 输入文件夹路径
input_folder = r"f:\04全球人口数据_3\_dpyear_1km"
# 输出文件夹路径
output_folder = r"f:\041_3\YBHM_env_1971_2020"
os.makedirs(output_folder, exist_ok=True)

# 目标范围 (minx, miny, maxx, maxy)
target_extent = (72.3, 21, 98.5, 32.6)  # 72.3°E-98.5°E, 21°N-32.6°N

# 获取所有tif文件
tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

# 1. 检查并统一坐标系
cropped_files = []
for tif in tif_files:
    input_path = os.path.join(input_folder, tif)
    output_path = os.path.join(output_folder, f"cropped_{tif}")

    # 检查原始坐标系
    src_ds = gdal.Open(input_path)
    src_proj = src_ds.GetProjection()
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_proj)
    print(f"文件 {tif} 的坐标系: {src_srs.GetAttrValue('PROJCS') or src_srs.GetAttrValue('GEOGCS')}")

    # 统一转换到WGS84并裁剪
    ds = gdal.Warp(output_path,
                  input_path,
                  dstSRS='EPSG:4326',  # 统一转换到WGS84
                  outputBounds=target_extent,
                  outputBoundsSRS='EPSG:4326',
                  format='GTiff',
                  dstNodata=0,
                  xRes=0.01,  # 设置x方向分辨率为0.01度
                  yRes=0.01)  # 设置y方向分辨率为0.01度
    ds = None
    src_ds = None
    cropped_files.append(output_path)

# 2. 合并所有裁剪后的单波段文件为一个多波段文件
if cropped_files:
    # 读取第一个文件获取参考信息
    first_ds = gdal.Open(cropped_files[0])
    xsize = first_ds.RasterXSize
    ysize = first_ds.RasterYSize
    projection = first_ds.GetProjection()
    geotransform = first_ds.GetGeoTransform()
    first_ds = None

    # 创建多波段输出文件
    output_multi = os.path.join(output_folder, "pop_YBHM_1971_2020.tif")
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_multi, xsize, ysize, len(cropped_files), gdal.GDT_Float32)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)

    # 将每个单波段文件写入多波段文件
    for i, tif in enumerate(cropped_files, start=1):
        ds = gdal.Open(tif)
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        out_band = out_ds.GetRasterBand(i)
        out_band.WriteArray(data)
        out_band.SetDescription(os.path.basename(tif))  # 设置波段名称
        ds = None

    out_ds.FlushCache()
    out_ds = None

print("处理完成！裁剪后的文件保存在:", output_folder)
print("多波段文件已创建:", os.path.join(output_folder, "pop_YBHM_1971_2020.tif"))