# 城镇面积数据通过csv文件进行插值计算补全空缺年份，仅插值到2020年

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np


def interpolate_years(input_csv, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)

    # 提取所有年份列并过滤掉2025和2030年
    year_cols = [col for col in df.columns if col.startswith('E_') and int(col[2:]) <= 2020]
    years = sorted([int(col[2:]) for col in year_cols])

    # 生成需要插值的年份序列（到2020年为止）
    full_years = range(min(years), 2021)  # 2021不包括

    # 对每个像元进行插值
    for year in full_years:
        if year not in years:
            # 找出前后最近的年份
            prev_year = max(y for y in years if y < year)
            next_year = min(y for y in years if y > year)

            # 计算插值权重
            weight = (year - prev_year) / (next_year - prev_year)

            # 进行线性插值
            prev_col = f"E_{prev_year}"
            next_col = f"E_{next_year}"
            df[f"E_{year}"] = df[prev_col] + (df[next_col] - df[prev_col]) * weight

    # 按年份排序所有列（只保留到2020年）
    sorted_cols = ['lon', 'lat'] + sorted([col for col in df.columns if col.startswith('E_') and int(col[2:]) <= 2020],
                                          key=lambda x: int(x[2:]))
    df = df[sorted_cols]

    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"插值完成，结果已保存到 {output_csv}")


if __name__ == "__main__":
    input_csv = "f:\\041_4\\GHS_B_S_csv\\urban_area_data.csv"
    output_csv = "f:\\041_4\\GHS_B_S_csv\\urban_area_data_interpolated.csv"
    interpolate_years(input_csv, output_csv)