import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def analyze_dp_distribution(req_start_info: list, save_dir: str = 'plot'):
    """
    req_start_info: List of (start_time, dp_rank, req_name)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    req_start_info.sort(key=lambda x: x[0])
    
    times = [x[0] for x in req_start_info]
    ranks = [x[1] for x in req_start_info]
    
    min_time = min(times)
    times = [t - min_time for t in times]

    jitter_y = np.random.uniform(-0.15, 0.15, size=len(ranks))
    
    # X轴抖动：微小的左右浮动，避免同一个 Batch 的点垂直连成一条线，稍微错开一点更易读
    # 根据你的时间跨度（0-2.5s），X轴抖动设为 0.01s 比较合适
    jitter_x = np.random.uniform(-0.01, 0.01, size=len(times))

    # --- 1. 散点图 ---
    plt.figure(figsize=(14, 7)) #稍微把图拉宽一点
    
    # 使用带抖动的坐标画图
    # alpha=0.5: 半透明，这样重叠越多的地方颜色越深
    # s=10: 点的大小
    scatter = plt.scatter(times + jitter_x, ranks + jitter_y, 
                            alpha=0.5, s=15, c=ranks, cmap='tab10', edgecolors='none')
    
    plt.title(f'Request Dispatching Pattern (n={len(ranks)}) - with Jitter')
    plt.xlabel('Time (s)')
    plt.ylabel('DP Rank ID')
    
    # 强制 Y 轴显示整数刻度，但加一点余量防止抖动出界
    plt.yticks(sorted(list(set(ranks))))
    plt.ylim(min(ranks) - 0.5, max(ranks) + 0.5) 
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.grid(axis='y', linestyle='-', alpha=0.2)
    
    save_path = os.path.join(save_dir, 'dp_dispatch_scatter_jitter.png')
    plt.savefig(save_path, dpi=150)
    print(f"[Plot] Saved Jittered scatter plot to {save_path}")
    plt.close()
    
def plot_x_vs_time(x, Y, save_path='batch_time_relation.png'):    
    # 2. 转换为 DataFrame，方便 Seaborn 绘图
    df = pd.DataFrame({
        'Batch Size': x,
        'Time (ms)': Y
    })

    plt.figure(figsize=(10, 6))
    
    # 3. 绘制散点图
    # x轴为 Batch, y轴为 Time
    sns.scatterplot(data=df, x='Batch Size', y='Time (ms)', 
                    color='royalblue', s=100, alpha=0.7, edgecolor='k')
    
    # (可选) 如果你想添加一条线性回归趋势线，可以取消下面这行的注释
    # sns.regplot(data=df, x='Batch Size', y='Time (ms)', scatter=False, color='red', line_kws={'linestyle':'--'})

    plt.title('Relationship between Batch Size and Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_and_save_distribution(data_list, save_path='1.png'):        
    plt.figure(figsize=(8, 6))
    sns.histplot(data_list, kde=True, color='skyblue', edgecolor='black')
                                                
    plt.title('Distribution Plot')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
                                                
    plt.savefig(save_path)
    plt.close()
