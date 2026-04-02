import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # 别忘了导入 pandas
import numpy as np

# 设置风格
sns.set_style("whitegrid")
sns.set_palette("deep")

# 1. 加载数据 (模拟数据加载，保留你的函数)
def load_list(filename):
    # 这里假设文件存在，实际运行时请确保路径正确
    with open(filename, 'r') as f:
        return json.load(f)

list1 = load_list('avail_tokens.json')
list2 = load_list('sim_avail_tokens.json')
min_len = min(len(list1), len(list2))

# 1. 假设这是你的数据 (替换为你真实的 DataFrame)
# df = pd.read_csv("your_log.csv") 
# 结构应该是: ['Iteration', 'Available Tokens', 'framework']

# --- 模拟生成一些数据用于演示 ---
iterations = np.arange(0, min_len)

df_sglang = pd.DataFrame({'Iteration': iterations, 'Available Tokens': list1[:min_len], 'framework': 'SGLang'})
df_ross = pd.DataFrame({'Iteration': iterations, 'Available Tokens': list2[:min_len], 'framework': 'ROSS'})
df = pd.concat([df_sglang, df_ross])
# -----------------------------

# 2. 关键步骤：数据分桶 (Binning)
# 将 Iteration 每 500 步归为一个桶，计算平均值
bin_size = 500
df['Interval'] = (df['Iteration'] // bin_size) * bin_size

# 聚合数据：计算每个区间的平均显存
df_agg = df.groupby(['Interval', 'framework'])['Available Tokens'].mean().reset_index()

# 3. 绘图
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# 使用 barplot，x轴是区间，hue区分框架
ax = sns.barplot(
    data=df_agg, 
    x='Interval', 
    y='Available Tokens', 
    hue='framework',
    palette={'SGLang': '#4c72b0', 'ROSS': '#dd8452'}, # 保持之前的蓝橙配色
    alpha=0.9
)

# 4. 美化
plt.title(f'Available Tokens By Iteration (Avg per {bin_size})', fontsize=14)
plt.xlabel('Iteration Interval', fontsize=12)
plt.ylabel('Avg Available Tokens', fontsize=12)
plt.legend(title='Framework')

# 格式化 Y 轴为 M (百万)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.1f}M'.format(x/1e6) for x in current_values])

plt.tight_layout()
plt.savefig('plot/avail_tokens.png')