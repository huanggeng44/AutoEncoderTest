import cdsapi
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# 初始化CDS API客户端
client = cdsapi.Client()
dataset = "cems-glofas-historical"
# 定义请求参数模板
request_template = {
    "system_version": ["version_4_0"],
    "hydrological_model": ["lisflood"],
    "product_type": ["consolidated"],
    "variable": ["river_discharge_in_the_last_24_hours"],
    "hmonth": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "hday": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30",
        "31"
    ],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [33, 72, 21, 99]  # N, W, S, E
}

# 定义起始和结束年份
start_year = 2010
end_year = 2012

# 循环遍历每个年份并下载数据
for year in range(start_year, end_year + 1):
    print(f"正在下载{year}年的数据...")

    # 复制请求模板以避免参数被覆盖
    request = request_template.copy()

    # 设置年份参数（假设'hyear'表示水文年，如果需要日历年，请根据CDS API文档调整）
    request["hyear"] = [str(year)]

    # 定义输出文件路径
    filename = f"F:/01water_4/1/gev_discharge_{year}.zip"

    try:
        # 发起数据下载请求
        client.retrieve(
            dataset,
            request=request,
            target=filename
        )
        print(f"{year}年的数据已成功下载到 {filename}")
    except Exception as e:
        print(f"下载{year}年数据时出错: {e}")

print("所有数据下载完成。")