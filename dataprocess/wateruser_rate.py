import pandas as pd
from collections import defaultdict
import numpy as np




def compare_headers(df1, df2):
    """生成表头差异报告"""
    diff_report = defaultdict(dict)

    # 标记存在性差异
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    diff_report['only_in_df1'] = sorted(df1_cols - df2_cols)
    diff_report['only_in_df2'] = sorted(df2_cols - df1_cols)

    # 标记顺序差异
    common_cols = sorted(df1_cols & df2_cols)
    order_diff = [(i, c) for i, c in enumerate(df1.columns) if c in common_cols] != \
                 [(i, c) for i, c in enumerate(df2.columns) if c in common_cols]

    # 生成报告
    report = {
        "total_df1": len(df1.columns),
        "total_df2": len(df2.columns),
        "common_cols": len(common_cols),
        "order_diff": order_diff,
        "col_details": diff_report
    }
    return report


def print_report(report):
    """可视化打印差异报告"""
    print(f"文件1总列数: {report['total_df1']}")
    print(f"文件2总列数: {report['total_df2']}")
    print(f"共有列数量: {report['common_cols']}")
    print(f"列顺序差异: {'存在' if report['order_diff'] else '无'}")

    # 打印特有列信息
    if report['col_details']['only_in_df1']:
        print("\n文件1特有列(前10个):")
        print(report['col_details']['only_in_df1'][:10])
    if report['col_details']['only_in_df2']:
        print("\n文件2特有列(前10个):")
        print(report['col_details']['only_in_df2'][:10])


# 使用示例
if __name__ == "__main__":
    # 读取文件（示例路径）
    # 读取两个CSV文件
    df1 = pd.read_csv(r'C:\Users\83403\Desktop\YB_basin_wateruse\2_cons_irr_YBHM.csv')
    # df2 = pd.read_csv(r'C:\Users\83403\Desktop\YB_basin_wateruse\2_cons_liv_YBHM.csv')
    df2_1_path = r'C:\Users\83403\Desktop\YB_basin_wateruse\2_cons_liv_YBHM_1'
    # # 列名清洗函数
    # def clean_columns(columns):
    #     """处理从第4列开始的列名（索引3）"""
    #     return [col[:-1] if i >= 3 and col.endswith(' ') else col
    #             for i, col in enumerate(columns)]
    #
    #
    # # 执行列名清洗（安全版）
    # try:
    #     original_cols = df2.columns.tolist()
    #     new_columns = clean_columns(original_cols)
    #
    #     # 验证列名唯一性
    #     if len(new_columns) != len(set(new_columns)):
    #         dupes = [col for col in new_columns if new_columns.count(col) > 1]
    #         raise ValueError(f"列名重复: {set(dupes)}")
    #
    #     df2.columns = new_columns
    # except Exception as e:
    #     print("清洗失败，回滚列名")
    #     df2.columns = original_cols
    #     raise
    #
    # # 验证清洗结果（示例输出）
    # print("清洗前后列名对比：")
    # for o, n in zip(original_cols[2:5], new_columns[2:5]):
    #     print(f"{o} → {n}")
    #
    # # 保存修改后的文件（可选）
    # df2.to_csv(df2_1_path, index=False)

    df2_1 = pd.read_csv(df2_1_path)

    # # 生成报告
    # report = compare_headers(df1, df2_1)
    # print_report(report)
    #
    # # 自动化处理建议
    # if report['common_cols'] == 0:
    #     print("\n严重错误：两文件没有共有列！")
    # elif report['common_cols'] < min(report['total_df1'], report['total_df2']):
    #     print("\n警告：存在列名差异，建议处理以下情况:")
    #     # 时间列格式自动检测
    #     time_cols1 = [c for c in df1.columns if c not in ['lon', 'lat']]
    #     time_cols2 = [c for c in df2_1.columns if c not in ['lon', 'lat']]
    #     if len(time_cols1) != len(time_cols2):
    #         print("检测到时间列数量不一致，请检查数据完整性")
    #     else:
    #         print("可能的时间列格式差异：")
    #         print(f"文件1示例: {time_cols1[:3]}")
    #         print(f"文件2示例: {time_cols2[:3]}")

    # # 手动指定月份列名模板（根据实际列名修改）
    # month_columns = [f"F{year}{month:02d}" for year in range(1971,2011) for month in range(1,13)]
    # # 验证列名有效性
    # valid_columns_df1 = [col for col in month_columns if col in df1.columns]
    # valid_columns_df2 = [col for col in month_columns if col in df2_1.columns]
    #
    # print(f"文件1有效列数：{len(valid_columns_df1)}")
    # print(f"文件2有效列数：{len(valid_columns_df2)}")
    #
    # # # 提取所有用水量月份列名（假设列名格式一致）
    # # water_columns = [col for col in df1.columns if col not in ['lon', 'lat']]
    # # 取两个文件共有的有效列
    # common_columns = list(set(valid_columns_df1) & set(valid_columns_df2))
    # common_columns.sort()  # 保持时间顺序
    # # 将第二个文件的水量数据乘以100
    # # df2[water_columns] *= 100
    # df2_1[common_columns] = df2_1[common_columns] / 100
    #
    # # 根据经纬度合并两个DataFrame，只保留共有的坐标点
    # merged_df = df1.merge(df2_1, on=['lon', 'lat'], suffixes=('', '_y'))
    # # 合并数据（使用outer保留所有点位）
    # merged_df = pd.merge(df1, df2_1, on=['lon', 'lat'], suffixes=('', '_y'), how='outer')
    # # 对每个月份列进行相加操作
    # for col in common_columns:
    #     merged_df[col] = merged_df[col].fillna(0) + merged_df.get(f'{col}_y', 0).fillna(0)
    #     if f'{col}_y' in merged_df:
    #         merged_df.drop(f'{col}_y', axis=1, inplace=True)
    # # 处理可能存在的单一文件独有点位
    # merged_df.fillna(0, inplace=True)
    # 保存结果到新CSV
    merged_irr_mon_path = r'C:\Users\83403\Desktop\YB_basin_wateruse\cons_aggri_YBHM.csv'
    year_irr_path = r'C:\Users\83403\Desktop\YB_basin_wateruse\cons_aggri_YBHM_year.csv'
    # # merged_df.to_csv(merged_mon_path, index=False)
    # # print(f"处理完成，结果已保存到cons_aggri_YBHM.csv,共合并{len(merged_df)}个点位数据")
    #================================================================================

    # # 读取原始文件（示例路径）
    # df_merge = pd.read_csv(merge_dom_mon_path)
    # # 列处理配置
    # YEAR_START = 1971
    # YEAR_END = 2010
    # VALUE_SCALE = 1000  # 缩放系数
    # MONTH_COL_START = 3  # 第4列（索引从0开始）
    #
    # # 生成年度列名
    # year_columns = [f"F{year}" for year in range(YEAR_START, YEAR_END+1)]
    #
    # # 验证数据完整性
    # total_month_columns = (YEAR_END - YEAR_START + 1) * 12
    # assert len(df_merge.columns[MONTH_COL_START:]) == total_month_columns, "月度数据列数不匹配"
    #
    # # 创建年度汇总DataFrame
    # annual_df = pd.DataFrame()
    # annual_df[df_merge.columns[:MONTH_COL_START]] = df_merge.iloc[:, :MONTH_COL_START]
    #
    # # 按年聚合数据
    # for i, year in enumerate(range(YEAR_START, YEAR_END+1)):
    #     start = MONTH_COL_START + i*12
    #     end = start + 12
    #     annual_df[year_columns[i]] = df_merge.iloc[:, start:end].sum(axis=1)
    #     # 添加检查确保实际年份匹配
    #     actual_year = int(df_merge.columns[start][1:5])  # 假设列名为F197101格式
    #     assert actual_year == year, f"年份不匹配: {actual_year} vs {year}"
    #
    # # 输出验证
    # print(f"原始月度列数: {len(df_merge.columns[MONTH_COL_START:])}")
    # print(f"生成年度列数: {len(year_columns)}")
    # print("前3行示例:")
    # print(annual_df.head(3))
    # # 保存结果
    # annual_df.to_csv(year_dom_path, index=False)
    #=============================================================================================
    merge_dom_mon_path = r"C:\Users\83403\Desktop\YB_basin_wateruse\2_cons_dom_YBHM.csv"
    year_dom_path = r"C:\Users\83403\Desktop\YB_basin_wateruse\cons_dom_YBHM_year.csv"
    # 读取数据（示例路径）
    df = pd.read_csv(merge_dom_mon_path)

    # ================= 参数配置 =================
    START_YEAR = 1971  # 数据起始年份
    END_YEAR = 2010  # 数据结束年份
    COLS_TO_PRESERVE = 3  # 需要保留的前导列数（lon, lat等）
    SCALE_FACTOR = 100000  # 数据缩放系数
    # =============================================

    # 数据校验
    expected_months = (END_YEAR - START_YEAR + 1) * 12
    actual_months = len(df.columns[COLS_TO_PRESERVE:])

    if actual_months != expected_months:
        raise ValueError(f"数据不完整！应有{expected_months}个月份，实际发现{actual_months}列")

    # 按年处理的核心逻辑
    annual_data = pd.DataFrame()
    for year in range(START_YEAR, END_YEAR + 1):
        # 计算列位置
        year_index = year - START_YEAR
        start_col = COLS_TO_PRESERVE + year_index * 12
        end_col = start_col + 12

        # 提取当年数据
        monthly_data = df.iloc[:, start_col:end_col]

        # 执行计算（年总和 + 缩放）
        annual_sum = monthly_data.sum(axis=1) / SCALE_FACTOR

        # 生成列名
        annual_data[f'F{year}'] = annual_sum

    # 构建最终结果
    result = pd.concat([
        df.iloc[:, :COLS_TO_PRESERVE],  # 保留原始列
        annual_data
    ], axis=1)

    # 结果验证
    print("处理完成！数据样例：")
    print(result[[
        result.columns[0],  # 显示第一列
        result.columns[1],  # 显示第二列
        'F1971',  # 第一年
        'F1980',  # 中间年份
        'F2010'  # 最后一年
    ]].head(3))

    # 保存结果（路径使用原始字符串）
    result.to_csv(year_dom_path, index=False)
    # =============================================================================================

    # 读取年度汇总数据（假设已按年聚合）
    df_Y = pd.read_csv(year_dom_path)

    # 配置参数
    START_YEAR = 1971
    END_YEAR = 2010
    YEARS = np.arange(START_YEAR, END_YEAR + 1)  # 1971-2010
    N = len(YEARS)  # 40年

    # 提取年度列（确保列顺序正确）
    year_cols = [f"F{year}" for year in YEARS]
    y_matrix  = df_Y[year_cols].values  # 转换为NumPy矩阵 (n_samples, n_years)

    # 向量化计算斜率（高效算法）
    def vectorized_slope(y):
        """矩阵版斜率计算，每行对应一个坐标点的时间序列"""
        # 优化后的斜率计算函数
        x = YEARS - YEARS.mean()  # 中心化处理
        sum_x2 = (x ** 2).sum()
        # 使用NumPy广播机制
        y_centered = y - y.mean(axis=1, keepdims=True)  # 正确使用keepdims
        sum_xy = np.dot(y_centered, x)
        return sum_xy / sum_x2

    # 执行计算
    slopes = vectorized_slope(y_matrix)

    # 创建结果DataFrame
    result = pd.DataFrame({
        'lon': df_Y['lon'],
        'lat': df_Y['lat'],
        'slope': slopes,
        'change_rate': slopes / y_matrix[:, 0]   * 100  # 相对于首年的变化率百分比
    })

    # 保存结果
    # water_usage_trend_path = r'C:\Users\83403\Desktop\YB_basin_wateruse\slope_aggri_YBHM_year.csv'
    water_usage_trend_path = r'C:\Users\83403\Desktop\YB_basin_wateruse\slope_dom_YBHM_year.csv'
    result.to_csv(water_usage_trend_path, index=False)

    # 验证示例
    print("前5个点的计算结果：")
    print(result.head())
    print(f"\n全局平均年变化量：{slopes.mean():.2f} 单位/年")
