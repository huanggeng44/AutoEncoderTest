import geopandas as gpd
import os

def combine_industrial_wateruse(input_files, output_path):
    """
    合并多个工业用水量shapefile数据
    
    参数:
        input_files: 输入shapefile路径列表
        output_path: 输出shapefile路径
    """
    # 读取第一个文件作为基础
    base_gdf = gpd.read_file(input_files[0])
    
    # 提取年份列名 (F1971-F2010)
    year_columns = [col for col in base_gdf.columns 
                   if col.startswith('F') and col[1:].isdigit()]
    
    # 创建结果GeoDataFrame，复制几何和其他非年份字段
    result_gdf = base_gdf.drop(columns=year_columns)
    
    # 初始化年份数据为0
    for year_col in year_columns:
        result_gdf[year_col] = 0.0
    
    # 遍历所有输入文件并累加用水量
    for file_path in input_files:
        gdf = gpd.read_file(file_path)
        for year_col in year_columns:
            result_gdf[year_col] += gdf[year_col]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果
    result_gdf.to_file(output_path, encoding='utf-8')
    print(f"合并后的工业总用水量数据已保存至: {output_path}")

if __name__ == "__main__":
    # 输入文件路径
    input_files = [
        r"g:\Projects\Test\my-project\wateruse_datasets\5_cons_elec_YBHM_year.shp",
        r"g:\Projects\Test\my-project\wateruse_datasets\5_cons_mfg_YBHM_year.shp", 
        r"g:\Projects\Test\my-project\wateruse_datasets\5_cons_min_YBHM_year.shp"
    ]
    
    # 输出文件路径
    output_path = r"g:\Projects\Test\my-project\wateruse_outputs\cons_industrial_total.shp"
    
    # 执行合并
    combine_industrial_wateruse(input_files, output_path)