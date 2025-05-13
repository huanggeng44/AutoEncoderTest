import geopandas as gpd
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams, ticker
import numpy as np
from matplotlib.font_manager import FontProperties

def plot_wateruse_trends(shp_path, output_dir, wateruse_type, wateruse_type_CHN):
    """
    读取shapefile中的流域用水数据并绘制折线图
    
    参数:
        shp_path: shapefile文件路径
        output_dir: 输出目录路径
        wateruse_type: 用水类型英文标识
        wateruse_type_CHN: 用水类型中文名称
    """
    # 设置全局字体（数字/字母用Times New Roman）
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.unicode_minus'] = False
    
    # 创建中文字体（黑体）
    hei = FontProperties(
        fname=r'C:\Windows\Fonts\simhei.ttf',
        size=16
    )
    # 创建中文字体属性（宋体）
    song = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=12)  # Windows字体路径

    # 读取shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 提取年份列名 (F1971-F2010)
    year_columns = [col for col in gdf.columns if col.startswith('F') and col[1:].isdigit()]
    years = [int(col[1:]) for col in year_columns]
    
    # 创建图形
    plt.figure(figsize=(15, 9))
    
    # 生成40种独特颜色和线型组合
    cmap1 = plt.get_cmap('tab20', 20)
    cmap2 = plt.get_cmap('tab20b', 20)  # 从10增加到20
    colors = list(cmap1(np.arange(20))) + list(cmap2(np.arange(20)))
    line_styles = [(0, ()), (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1))]  # 4种线型

    # 为每个流域绘制折线
    for idx, row in gdf.iterrows():
        basin_name = row['Basin_Name']
        water_use = [row[col] for col in year_columns]
        style = {
            'color': colors[idx % 40],  # 使用40种颜色循环
            'linestyle': line_styles[idx//10 % 4],  # 4种线型
            'linewidth': 0.8 + (idx % 3) * 0.2,
            'marker': 'o',  # 固定使用圆形标记
            'markersize': 4
        }
        plt.plot(years, water_use, label=basin_name, **style)
    
    # 设置图形属性
    plt.title(f'1971-2010年各流域{wateruse_type_CHN}年用水量变化', fontproperties=hei, fontsize=18, pad=20)
    plt.xlabel('年份', fontproperties=song, fontsize=18, labelpad=10)
    
    # 获取当前轴并设置科学计数法格式
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)  # 禁用偏移量显示
    
    # 动态计算科学计数法的乘数并添加到ylabel
    y_data = np.concatenate([row[year_columns].values for _, row in gdf.iterrows()])
    max_val = np.max(y_data)
    exponent = int(np.log10(max_val)) if max_val > 0 else 0
    if exponent != 0:
        plt.ylabel(f'{wateruse_type_CHN}用水量(10^{exponent} km³)', 
                  fontproperties=song, fontsize=18, labelpad=10)
    else:
        plt.ylabel(f'{wateruse_type_CHN}用水量(km³)', 
                  fontproperties=song, fontsize=18, labelpad=10)
    
    # 网格线设置（仅保留横向虚线）
    plt.grid(True, axis='y', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # 横轴刻度设置
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xticks(ha='center', va='top', fontsize=18)
    plt.yticks(fontsize=18)
    
    # 设置图例
    legend = ax.legend(
        title="流域名称",
        title_fontproperties=hei,
        bbox_to_anchor=(1, 1),
        loc="upper left",
        frameon=False,
        ncol=1,
        fontsize=11,
        borderaxespad=0.3,
        columnspacing=0.8,
        handlelength=2.2
    )
    legend.get_title().set_ha("left")
    legend._legend_box.align = "left"
    legend._legend_title_box.sep = 20
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    
    # 保存图形
    output_path = os.path.join(output_dir, f'{wateruse_type}_wateruse_trends.png')
    plt.savefig(output_path, dpi=900, bbox_inches='tight')
    print(f"图形已保存至: {output_path}")
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    # 输入文件路径 (相对路径)
    wateruse_type = 'cons_elec'  # 流域类型
    wateruse_type_CHN = '工业'   # 流域类型中文
    
    shp_file = os.path.join('..', 'wateruse_datasets', f'5_{wateruse_type}_YBHM_year.shp')
    
    # 输出目录 (相对路径)
    output_directory = os.path.join('..', 'wateruse_outputs')
    
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)
    
    # 绘制图形
    plot_wateruse_trends(shp_file, output_directory, wateruse_type, wateruse_type_CHN)