
# 读取nc文件
import os
import warnings
os.environ['PROJ_LIB'] = r'C:\Users\83403\AppData\Local\pypoetry\Cache\virtualenvs\my-project-foBy-FZ1-py3.12\Lib\site-packages\osgeo\data\proj'
from osgeo import gdal,osr,ogr
from netCDF4 import Dataset
import numpy as np
import rasterio
from rasterio.transform import from_origin


def convert_nc_to_tif(nc_file, tif_file, variable_name):
    # Open NetCDF file
    nc_dataset = Dataset(nc_file, 'r')
    # Extract the variable data
    var = nc_dataset.variables[variable_name]
    variable_data = nc_dataset.variables[variable_name][:]
    # 读取变量数据并处理 NaN 值
    no_data_value = -9999  # 选择一个适当的 NoData 值
    data_filled = np.where(np.isnan(variable_data), no_data_value, variable_data)
    data_filled = data_filled.astype(var.dtype)  # 保持原有数据类型
    # Get geospatial information
    lon = nc_dataset.variables['longitude'][:]
    lat = nc_dataset.variables['latitude'][:]
    lon_min, lon_max, lat_min, lat_max = lon.min(), lon.max(), lat.min(), lat.max()
    # 左上角坐标，XY像元大小
    X_left_top = lon_min - 0.025
    Y_left_top =  lat_max + 0.025
    X_cell_size = ((lon_max + 0.025) - (lon_min - 0.025)) / len(lon)
    Y_cell_size = -((lat_max + 0.025) - (lat_min - 0.025)) / len(lat)

    # rasterio
    transform = from_origin(X_left_top, Y_left_top, X_cell_size, -Y_cell_size)
    with rasterio.open(
            tif_file,
            'w',
            driver='GTiff',
            width=data_filled.shape[2],
            height=data_filled.shape[1],
            count=data_filled.shape[0] if len(data_filled.shape) > 2 else 1,  # 多波段输出
            dtype=data_filled.dtype,
            crs='EPSG:4326',  # 根据实际情况设置投影信息
            transform=transform,
            nodata=no_data_value,
    ) as dst:
        if len(data_filled.shape) > 2:
            for band in range(1, data_filled.shape[0] + 1):
                dst.write(data_filled[band - 1, :, :], band)
        else:
            dst.write(data_filled, 1)



    # # Create GeoTIFF(gdal)
    # driver = gdal.GetDriverByName('GTiff')
    # tif_dataset = driver.Create(tif_file, len(lon), len(lat), len(variable_data), gdal.GDT_Float32)
    # # Set GeoTIFF geospatial information
    # tif_dataset.SetGeoTransform([X_left_top, X_cell_size, 0,Y_left_top, 0, Y_cell_size])
    #
    # tif_dataset.SetProjection('EPSG:4326') # Assuming WGS 84
    # # Write variable data to GeoTIFF
    # for i in range(len(variable_data)):
    #     tif_dataset.GetRasterBand(i + 1).WriteArray(variable_data[i])
    # # tif_dataset.GetRasterBand(1).WriteArray(variable_data[0])
    # # Close datasets
    # tif_dataset = None
    # nc_dataset = None



if __name__ == "__main__":

    nc_file_path = r'F:/01water_4/2015_A_global_streamflow_reanalysis/2015_consolidated.nc'

    # nc_dataset = Dataset(nc_file_path, 'r')
    # # Extract the variable data
    # variable_names = nc_dataset.variables.keys()
    # print("NetCDF文件中的变量名称如下：")
    # for var_name in variable_names:
    #     print(var_name)
    # # NetCDF文件中的变量名称如下：
    # # valid_time
    # # surface
    # # latitude
    # # longitude
    # # dis24
    #
    # # 关闭文件
    # nc_dataset.close()

    # nc转tiff文件
    nc_to_tif_file_path = f"F:/011_4/nc_to_tiff/streamflow_2015_1.tif"
    variable_name = "dis24"
    convert_nc_to_tif(nc_file_path, nc_to_tif_file_path, variable_name)
