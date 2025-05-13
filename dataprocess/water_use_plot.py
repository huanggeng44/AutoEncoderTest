import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap

# 设置全局字体（数字/字母用Times New Roman）
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 创建中文字体属性（宋体）
song = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=12)  # Windows字体路径
# 创建加粗版的中文字体配置（在原有代码基础上修改）
song_bold = FontProperties(
    fname=r'C:\Windows\Fonts\simsun.ttc',  # 字体路径与之前一致
    size=12,                               # 字号保持原样
    weight='bold'                          # 关键参数：加粗
)
# 创建黑体字体对象（Windows系统路径）
hei = FontProperties(
    fname=r'C:\Windows\Fonts\simhei.ttf',  # 黑体字体文件
    size=16
)

# 生成30种独特颜色（扩展色域）
cmap1 = plt.get_cmap('tab20', 20)  # 前20种颜色
cmap2 = plt.get_cmap('tab20b', 10)  # 后10种颜色
colors = list(cmap1(np.arange(20))) + list(cmap2(np.arange(10)))
# 定义6种线型模式（包含虚实组合）
line_styles = [
    (0, ()),  # 实线 (solid)
    (0, (5, 2)),  # 虚线 (dashed)
    (0, (3, 1, 1, 1))  # 点划线 (dash-dot)
]
# 生成样式组合（颜色+线型）
styles = [
    {
        'color': colors[i],
        'linestyle':  line_styles[i//10],  # 每5种颜色换一次线型
        'linewidth': 0.8 + (i % 3) * 0.2,  # 添加细微线宽变化
        'alpha': 0.9 if i < 15 else 0.8  # 透明度微调
    }
    for i in range(30)
]

# 读取Excel的第二个Sheet（索引从0开始，第二个Sheet索引为1）
df = pd.read_excel(r"C:\Users\83403\Desktop\YB_basin_wateruse\three_sectors_cons.xlsx", sheet_name=2)
# df = pd.read_excel(r"C:\Users\83403\Desktop\YB_basin_wateruse\three_sectors_cons.xlsx", sheet_name=3)

# 提取年份列名（假设前三列是FID和basin_name，之后为年份列）
year_columns = df.columns[3:]
years = [int(col[-4:]) for col in year_columns]  # 提取列名最后4位转为整数

# 设置画布大小
plt.figure(figsize=(15, 9))

# 绘制每个流域的折线
for idx, row in df.iterrows():
    basin_name = row["Basin_Name"]
    water_usage = row[year_columns].values
    plt.plot(
        years,
        water_usage,
        label=basin_name,
        marker='o',
        markersize=4,
        # linewidth=1.2,
        # alpha=0.8,
        **styles[idx]
    )

# 设置图表属性
plt.xlabel("年份", fontproperties=song,fontsize=18, labelpad=10)
plt.ylabel("农业用水量(km³)", fontproperties=song,fontsize=18, labelpad=10)
# plt.title("1971-2010年各流域用水量变化",fontproperties=hei, fontfamily='serif', fontsize=16, pad=20)
# plt.grid(True, linestyle='--', alpha=0.7)
# 网格线设置（仅保留横向虚线）
plt.grid(True,
         axis='y',
         linestyle='--',
         alpha=0.5,
         linewidth=0.8)

# 横轴刻度设置
ax = plt.gca()
# 设置主刻度为每5年
ax.xaxis.set_major_locator(ticker.MultipleLocator(5)) # 每5年显示一个刻度
plt.xticks(ha='center', va='top', fontsize=18)       # 数字用Times New Roman
plt.yticks(fontsize=18)


# 设置图例（中文用宋体）
legend = ax.legend(
    title="流域名称",              # 图例标题文本
    title_fontproperties=hei,     # 设置标题字体
    bbox_to_anchor=(1.02,0.35),      # 图例框定位坐标
    loc="upper left",              # 定位基准点对齐方式
    frameon=False,                 # 边框显示控制
    ncol=1,                        # 列数设置
    fontsize=11,                    # 条目字号
    borderaxespad=0.3,            # 减小轴与图例间距
    columnspacing=0.8,           # 列间距（多列时适用）
    handlelength=3.2,             # 图标长度
    framealpha = 0.9
)
# 强制计算布局
plt.draw()

# 调整标题对齐方式（核心代码）
legend.get_title().set_ha("left")          # 文本左对齐
legend._legend_box.align = "left"          # 容器左对齐
# legend._legend_box.sep = 5                 # 主容器整体间距
legend._legend_title_box.sep = 20           # 标题与内容间距设为0
legend._legend_title_box.pad = 0           # 标题内边距设为0
# legend._legend_title_box.set_offset((0,0)) # 消除额外偏移
# 可选：调整标题垂直对齐
legend.get_title().set_va("bottom")  # 标题底部对齐

# 微调边距（可选）
# legend._legend_box.margin = 0              # 容器边距归零
# 调整边距
plt.tight_layout()
plt.subplots_adjust(right=0.78)

# 自动调整布局并显示
plt.tight_layout()
plt.show()