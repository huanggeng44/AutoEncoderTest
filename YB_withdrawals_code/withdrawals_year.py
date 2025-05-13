# 处理年csv数据的表头格式
# 计算各河段的2010年各部门的取水量

import pandas as pd
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def process_header(file_path, output_path):
    # 定义表头
    headers = ["Grid_ID", "lon", "lat", "ilon", "ilat"] + \
              [f"{year}_Y" for year in range(2010, 2105, 5)]

    # 读取CSV文件
    df = pd.read_csv(file_path, header=None, names=headers)

    # 可选：将科学计数法转换为普通浮点数（如果需要）
    for col in headers[5:]:  # 只转换年份数据列
        df[col] = df[col].astype(float)

    # 保存处理后的文件

    df.to_csv(output_path, index=False)

    print(f"文件已处理并保存到: {output_path}")

def watershed_withd_year(watershed, csv_files):
    # 3. 处理每个CSV文件
    for sector, file_path in csv_files.items():
        # 读取CSV数据
        df = pd.read_csv(file_path, encoding='utf-8')

        # 创建几何点
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

        # 空间连接
        joined = gpd.sjoin(gdf, watershed, how='inner', predicate='within')

        # 按流域分组并计算2010年总和
        result = joined.groupby('stations')['2010_Y'].sum().reset_index()
        result.columns = ['stations', f'2010_{sector}']

        # 保存结果
        output_path = f'f:/023_7/withdrawals_sectors_year/year_watershed_2010/{sector}_by_watershed.csv'
        result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("处理完成！所有部门的2010年取水数据已按流域统计并保存。")

if __name__ == "__main__":

    # file_path = "f:/023_7/withdrawals_sectors_year/year_csv/wdelec_km3peryr.csv"
    # header_path = "f:/023_7/withdrawals_sectors_year/year_csv_header/wdelec_km3peryr.csv"
    # process_header(file_path, header_path)

    # 1. 读取流域面数据
    watershed = gpd.read_file('f:/023_7/watershed_shp/Watershed_downriver_to_sea_wgs84.shp',encoding='utf-8')

    # 2. 定义要处理的CSV文件列表
    csv_files = {
        'dom': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wddom_km3peryr.csv',
        'elec': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wdelec_km3peryr.csv',
        'irr': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wdirr_km3peryr.csv',
        'liv': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wdliv_km3peryr.csv',
        'mfg': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wdmfg_km3peryr.csv',
        'min': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wdmin_km3peryr.csv',
        'total': 'f:/023_7/withdrawals_sectors_year/year_csv_header/wdtotal_km3peryr.csv'
    }
    watershed_withd_year(watershed, csv_files)

