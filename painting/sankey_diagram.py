#
#
# import plotly.graph_objects as go
#
# # 定义所有节点（包含所有层级的唯一名称）
# nodes = ["A1", "A2",  # Layer 0
#          "B1", "B2",  # Layer 1
#          "C1", "C2"]  # Layer 2
#
# # 定义连接关系：source → target
# sources = [0, 0, 1, 1,  # Layer 0 → Layer 1 (A1→B1, A1→B2, A2→B1, A2→B2)
#            2, 2, 3, 3]  # Layer 1 → Layer 2 (B1→C1, B1→C2, B2→C1, B2→C2)
# targets = [2, 3, 2, 3,  # Layer 0 → Layer 1 的目标索引
#            4, 5, 4, 5]  # Layer 1 → Layer 2 的目标索引
# values =  [8, 2, 4, 1,  # Layer 0 → Layer 1 的流量值
#            6, 2, 3, 1]  # Layer 1 → Layer 2 的流量值
#
# # 创建桑基图
# fig = go.Figure(go.Sankey(
#     node=dict(
#         pad=20,
#         thickness=20,
#         label=nodes,
#         # 分层显示：设置节点的 x/y 坐标（0-1之间）
#         x=[0.0, 0.0,    # Layer 0 (A1, A2)
#            0.3, 0.3,    # Layer 1 (B1, B2)
#            1, 1],   # Layer 2 (C1, C2)
#         y=[0.2, 0.8,    # Layer 0 的垂直位置
#            0.3, 1,    # Layer 1
#            0.4, 1],   # Layer 2
#     ),
#     link=dict(
#         source=sources,
#         target=targets,
#         value=values,
#         # 可选：设置不同层级的颜色
#         color=["rgba(255,0,0,0.3)", "rgba(0,255,0,0.3)",
#                "rgba(0,0,255,0.3)", "rgba(255,255,0,0.3)",
#                "rgba(128,0,128,0.3)", "rgba(0,128,128,0.3)",
#                "rgba(128,128,0,0.3)", "rgba(64,64,64,0.3)"]
#     )
# ))
#
# # 设置布局
# fig.update_layout(
#     title_text="多级桑基图示例",
#     font_size=12,
#     height=600
# )
#
# fig.show()

import pandas as pd
import plotly.graph_objects as go

file_path = r"G:\BasicDatasets\water_traj_similar_basins\traj_water_groups_pop.xlsx"
# 读取数据
df = pd.read_excel(file_path)


# 生成桑基图所需的数据结构
def prepare_sankey_data(df):
    headers = df.columns.tolist()
    print(headers)

    # 创建唯一节点列表（三级结构）
    # 定义自定义节点名称
    cluster_labels = [f"水文相似组{i + 1}" for i in range(8)]  # 8个聚类节点
    group_labels = ["用水量较大且稳定增长型", "用水量较少且缓慢增长型", "用水少且无明显变化","用水量大且快速增长型"]  # 4个组节点
    level_labels = ["＞100（百万）", "10~100（百万）", "1~10（百万）", "0.1~1（百万）"]  # 4个等级节点
    # 生成节点列表（三级结构）
    all_nodes  = cluster_labels + group_labels + level_labels
    node_indices = {node: i for i, node in enumerate(all_nodes)}

    # 创建链接数据
    links = []

    # # 第一层：聚类 -> 组
    # cluster_to_group = df.groupby([df.columns[3], df.columns[4]]).size().reset_index(name='count')
    # for _, row in cluster_to_group.iterrows():
    #     source = node_indices[cluster_labels[row.iloc[0]]]
    #     target = node_indices[group_labels[row.iloc[1]-1]]
    #     links.append({
    #         'source': source,
    #         'target': target,
    #         'value': row['count']
    #     })
    #
    # # 第二层：组 -> 人口等级
    # group_to_level = df.groupby([df.columns[4], df.columns[6]]).size().reset_index(name='count')
    # for _, row in group_to_level.iterrows():
    #     source = node_indices[group_labels[row.iloc[0]-1]]
    #     target = node_indices[level_labels[row.iloc[1]-1]]
    #     links.append({
    #         'source': source,
    #         'target': target,
    #         'value': row['count']
    #     })

    # 第一层链接：聚类 -> 组（强制排序）
    for cluster_idx in range(8):  # 按水文组1-8顺序遍历
        cluster_data = df[df.iloc[:, 3] == cluster_idx]  # 筛选当前水文组
        group_counts = cluster_data.iloc[:, 4].value_counts().sort_index()  # 按组编号排序

        for group_idx in range(1, 5):  # 遍历1-4组
            if group_idx in group_counts:
                links.append({
                    "source": node_indices[f"水文相似组{cluster_idx + 1}"],
                    "target": node_indices[group_labels[group_idx - 1]],
                    "value": int(group_counts[group_idx])
                })

    # 第二层链接：组 -> 等级（强制排序）
    for group_idx in range(4):  # 按组顺序遍历
        group_data = df[df.iloc[:, 4] == group_idx + 1]  # 筛选当前组
        level_counts = group_data.iloc[:, 5].value_counts().sort_index()  # 按等级排序

        for level in range(1, 5):  # 遍历1-4级
            if level in level_counts:
                links.append({
                    "source": node_indices[group_labels[group_idx]],
                    "target": node_indices[level_labels[level - 1]],
                    "value": int(level_counts[level])
                })

    return all_nodes, links


# 准备数据
nodes, links = prepare_sankey_data(df)

# 创建桑基图
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color=[f"hsl({i * 30},50%,50%)" for i in range(len(nodes))]  # 自动生成颜色
    ),
    link=dict(
        source=[l['source'] for l in links],
        target=[l['target'] for l in links],
        value=[l['value'] for l in links],
        color="rgba(150,150,150,0.3)"
    )
))

# 设置布局
fig.update_layout(
    title="用水模式桑基图分析",
    font=dict(size=12,  family="Arial"),
    height=800,
    margin=dict(t=60, l=50, r=50, b=50),
)

fig.show()

# # 保存为高分辨率图片
# fig.write_image(r"G:\BasicDatasets\water_traj_similar_basins\sankey.png",
#                 engine="kaleido",  # 必须的渲染引擎
#                 width=1200,       # 图片宽度
#                 height=900,       # 图片高度
#                 scale=2,         # 缩放系数（2=双倍分辨率）
#                 format="png")     # 支持png/jpeg/pdf等
#
# print("图片已保存为 sankey.png")