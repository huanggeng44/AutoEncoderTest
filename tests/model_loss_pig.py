import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 读取CSV文件，假设文件没有表头，自定义列名
# outputs_dir = r"G:\Projects\Test\cnn_transform_kaggle_project\cnn_soil_output"
# train_loss_path = r"G:\Projects\Test\cnn_transform_kaggle_project\cnn_soil_output\soil_training_loss_log_step_200.csv"
# test_loss_path = r"G:\Projects\Test\cnn_transform_kaggle_project\cnn_soil_output\soil_testing_loss_log_step_200.csv"
outputs_dir = r"G:\Projects\Test\my-project\outputs_FC"
train_loss_path = r"G:\Projects\Test\my-project\outputs_FC\train_losses.csv"
test_loss_path = r"G:\Projects\Test\my-project\outputs_FC\test_losses.csv"
df1 = pd.read_csv(train_loss_path, header=None, names=['epoch', 'train_loss'])
df2 = pd.read_csv(test_loss_path, header=None, names=['epoch', 'test_loss'])
df1_head = df1.columns.tolist()
df2_head = df2.columns.tolist()
# 或者读取后转换
# 强制转换为数值类型，无效值转为NaN
df1[["epoch", "train_loss"]] = df1[["epoch", "train_loss"]].apply(pd.to_numeric, errors="coerce")
df2[["epoch", "test_loss"]] = df2[["epoch", "test_loss"]].apply(pd.to_numeric, errors="coerce")



print(df1.dtypes)
print(df2.dtypes)

# 设置全局字体（中文用黑体，英文/数字用 Times New Roman）
mpl.rcParams['font.family'] = 'sans-serif'  # 默认字体类型
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文使用黑体
mpl.rcParams['font.serif'] = ['Times New Roman']  # 英文/数字使用 Times
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
plt.plot(df1["epoch"], df1["train_loss"], marker=None, linestyle='-', color='blue', label='训练损失')
plt.plot(df2["epoch"], df2["test_loss"], marker=None, linestyle='-', color='red', label='测试损失')
plt.xlabel('训练轮次')
plt.ylabel('损失值')
plt.title(f'流域时空特征提取自编码器模型训练与测试损失曲线')
# 设置刻度标签字体（强制指定 Times New Roman）
plt.xticks(fontproperties='Times New Roman', fontsize=10)
plt.yticks(fontproperties='Times New Roman', fontsize=10)
# 显示图例（中文）
plt.legend(prop={'family': 'SimHei', 'size': 10})
# 保存图像（支持中文路径）
fig_save_path_name = f"time_space_train_loss_per_epoch.png"
fig_save_path = os.path.join(outputs_dir, fig_save_path_name)
plt.savefig(fig_save_path, dpi=900, bbox_inches='tight')
plt.show()