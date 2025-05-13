# 处理雅布流域各河段的2010年逐月取水数据

import pandas as pd
import geopandas as gpd
import os

def read_csv_header(input_folder):
    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(input_folder, filename)

            # 读取CSV文件(只读取前几行获取字段名)
            try:
                df = pd.read_csv(filepath, nrows=1)
                print(f"\n文件: {filename}")
                print("字段列表:", list(df.columns))
                print("字段数量:", len(df.columns))

                # 显示前几行数据示例
                sample_df = pd.read_csv(filepath, nrows=3)
                # print("\n数据示例:")
                # print(sample_df)

            except Exception as e:
                print(f"处理文件 {filename} 时出错:", str(e))

    print("\n分析完成！")




def watershed_withd(input_csv, watershed, output_folder):
    # 3. 读取取水数据CSV
    df = pd.read_csv(input_csv, encoding='utf-8')

    # 4. 转换为GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs="EPSG:4326"
    )

    # 5. 空间连接
    joined = gpd.sjoin(gdf, watershed, how="inner", predicate="within")

    # 6. 提取月份列(格式为YYYYMM_m)
    month_cols = [col for col in df.columns if col.endswith('_m')]
    print(month_cols)

    # 7. 按流域分组统计
    result = joined.groupby('stations')[month_cols].sum().reset_index()

    # 8. 保存结果
    output_path = os.path.join(output_folder, "twdmin_km3permon_2010_watershed.csv")
    result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"处理完成！结果已保存到: {output_path}")

if __name__ == "__main__":
    # 设置输入文件夹路径
    # input_folder = r"f:\023_7\withdrawals_sectores_mon_2010\mon_csv"
    # read_csv_header(input_folder)

    # 1. 读取流域shapefile
    watershed = gpd.read_file(r"f:\023_7\watershed_shp\Watershed_downriver_to_sea_wgs84.shp",encoding='utf-8')

    # 2. 设置输入输出路径
    input_csv = r"f:\023_7\withdrawals_sectores_mon_2010\mon_csv\twdmin_km3permon_2010.csv"
    output_folder = r"f:\023_7\withdrawals_sectores_mon_2010\mon_watershed"
    os.makedirs(output_folder, exist_ok=True)
    watershed_withd(input_csv, watershed, output_folder)