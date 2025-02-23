import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

# filename = "f1_score_node_sketch"

# data = pd.read_csv(filename+'_GAT.csv')
# x1_GAT = np.arange(0, 40, 1)
# y1_GAT = data["FedAVG"][0:40]
# y2_GAT = data["FedSM_FT"][0:40]
# y3_GAT = data["FedMatch_FT"][0:40]
# y4_GAT = data["FLwF"][0:40]
# y5_GAT = data["FedMH(ours)"][0:40]
 

# x2_GAT = np.arange(40, 120, 1)
# y6_GAT = data["FedAVG"][40:]
# y7_GAT = data["FedSM_FT"][40:]
# y8_GAT = data["FedMatch_FT"][40:]
# y9_GAT = data["FLwF"][40:]
# y10_GAT = data["FedMH(ours)"][40:]

# data = pd.read_csv(filename+'_GCN.csv')

# x1_GCN = np.arange(0, 40, 1)
# y1_GCN = data["FedAVG"][0:40]
# y2_GCN = data["FedSM_FT"][0:40]
# y3_GCN= data["FedMatch_FT"][0:40]
# y4_GCN = data["FLwF"][0:40]
# y5_GCN = data["FedMH(ours)"][0:40]
 

# x2_GCN = np.arange(40, 120, 1)
# y6_GCN = data["FedAVG"][40:]
# y7_GCN = data["FedSM_FT"][40:]
# y8_GCN = data["FedMatch_FT"][40:]
# y9_GCN = data["FLwF"][40:]
# y10_GCN = data["FedMH(ours)"][40:]
# file_path_GCN = 'output/f1_scores_GCN.csv'
file_path_GCN_LSTM = 'output/f1_scores_GCN_LSTM.csv'
file_path_GAT = 'output/f1_scores_GAT.csv'
filename = 'GCN_cr1e-5_256_lr1e-4'
file_path_GCN = 'output/f1_scores_'+filename+'.csv'
rng = np.random.default_rng(42)
x1 = np.arange(0, 40, 1)
y1 = [0.01, 0.0225, 0.035, 0.0475, 0.06, 0.0725, 0.085, 0.0975, 0.11, 0.1225, 0.135, 0.1475, 0.16, 0.1725, 0.185, 0.1975, 0.21, 0.2225, 0.235, 0.2475, 0.26, 0.2725, 0.285, 0.2975, 0.31, 0.3225, 0.335, 0.3475, 0.36, 0.3725, 0.385, 0.3975, 0.41, 0.4225, 0.435, 0.4475, 0.46, 0.4725, 0.485, 0.4975]
y2 = [0.02, 0.033, 0.046, 0.059, 0.072, 0.085, 0.098, 0.111, 0.124, 0.137, 0.15, 0.163, 0.176, 0.189, 0.202, 0.215, 0.228, 0.241, 0.254, 0.267, 0.28, 0.293, 0.306, 0.319, 0.332, 0.345, 0.358, 0.371, 0.384, 0.397, 0.41, 0.423, 0.436, 0.449, 0.462, 0.475, 0.488, 0.501, 0.514, 0.527]
y3 = [0.03, 0.045, 0.06, 0.075, 0.09, 0.105, 0.12, 0.135, 0.15, 0.165, 0.18, 0.195, 0.21, 0.225, 0.24, 0.255, 0.27, 0.285, 0.3, 0.315, 0.33, 0.345, 0.36, 0.375, 0.39, 0.405, 0.42, 0.435, 0.45, 0.465, 0.48, 0.495, 0.51, 0.525, 0.54, 0.555, 0.57, 0.585, 0.6, 0.615]
y4 = [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82]
y5 = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.863, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]


x2 = np.arange(40, 120, 1)
y6 = 0.5 * rng.random(size=80) + 0.5
y7 = 0.5 * rng.random(size=80) + 0.5
y8 = 0.5 * rng.random(size=80) + 0.5
y9 = 0.5 * rng.random(size=80) + 0.5
y10 = 0.5 * rng.random(size=80) + 0.5

def read_csv_data(filename, column_name):
    data = pd.read_csv(filename)
    return data[column_name].values

# 读取数据
x1 = np.arange(0, 40, 1)
x2 = np.arange(40, 120, 1)

y1_FedAvg_GCN = read_csv_data(file_path_GCN, 'GCN_FedAvg')[:40]
y2_FedSem_FT_GCN = read_csv_data(file_path_GCN, 'GCN_FedSem_FT')[:40]
y3_FedMatch_FT_GCN = read_csv_data(file_path_GCN, 'GCN_FedMatch_FT')[:40]
y4_FLwF_GCN = read_csv_data(file_path_GCN, 'GCN_FLwF')[:40]
y5_FedMH_GCN = read_csv_data(file_path_GCN, 'GCN_FedMH')[:40]

y6_FedAvg_GCN = read_csv_data(file_path_GCN, 'GCN_FedAvg_incremental')
y7_FedSem_FT_GCN = read_csv_data(file_path_GCN, 'GCN_FedSem_FT_incremental')
y8_FedMatch_FT_GCN = read_csv_data(file_path_GCN, 'GCN_FedMatch_FT_incremental')
y9_FLwF_GCN = read_csv_data(file_path_GCN, 'GCN_FLwF_incremental')
y10_FedMH_GCN = read_csv_data(file_path_GCN, 'GCN_FedMH_incremental')


y1_FedAvg_GAT = read_csv_data(file_path_GAT, 'GAT_FedAvg')[:40]
y2_FedSem_FT_GAT = read_csv_data(file_path_GAT, 'GAT_FedSem_FT')[:40]
y3_FedMatch_FT_GAT = read_csv_data(file_path_GAT, 'GAT_FedMatch_FT')[:40]
y4_FLwF_GAT = read_csv_data(file_path_GAT, 'GAT_FLwF')[:40]
y5_FedMH_GAT = read_csv_data(file_path_GAT, 'GAT_FedMH')[:40]

y6_FedAvg_GAT = read_csv_data(file_path_GAT, 'GAT_FedAvg_incremental')
y7_FedSem_FT_GAT = read_csv_data(file_path_GAT, 'GAT_FedSem_FT_incremental')
y8_FedMatch_FT_GAT = read_csv_data(file_path_GAT, 'GAT_FedMatch_FT_incremental')
y9_FLwF_GAT = read_csv_data(file_path_GAT, 'GAT_FLwF_incremental')
y10_FedMH_GAT = read_csv_data(file_path_GAT, 'GAT_FedMH_incremental')

y1_FedAvg_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedAvg')[:40]
y2_FedSem_FT_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedSem_FT')[:40]
y3_FedMatch_FT_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedMatch_FT')[:40]
y4_FLwF_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FLwF')[:40]
y5_FedMH_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedMH')[:40]

y6_FedAvg_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedAvg_incremental')
y7_FedSem_FT_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedSem_FT_incremental')
y8_FedMatch_FT_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedMatch_FT_incremental')
y9_FLwF_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FLwF_incremental')
y10_FedMH_GCN_LSTM = read_csv_data(file_path_GCN_LSTM, 'GCN_LSTM_FedMH_incremental')
# 绘制图表
fig = plt.figure(figsize=(10, 12))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.5, 1])

gs1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[0], wspace=None, hspace=None, height_ratios=[1, 1, 1, 1])

gs2 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs0[1], wspace=None, hspace=None, height_ratios=[1, 1, 1, 1])


style = {
    'FedAvg': {'line':'solid', 'color':'gray'},
    'FedSem_FT': {'line':(0, (5, 1)), 'color':'navy'},
    'FedMatch_FT': {'line':'dotted', 'color':'orange'},
    'FLwF': {'line':(0, (3, 1, 1, 1)), 'color':'skyblue'},
    'FedMH (ours)': {'line':'dashdot', 'color':'red'},
    'linewidth': 3,
}

# ax1 = fig.add_subplot(gs1[0])
# ax1.set_title('Training Expert Model', fontweight='bold')
# ax1.text(-0.3, 0.95, "MLP", transform=ax1.transAxes, fontdict={'weight': 'bold'})
# ax1.plot(x1, y1, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
# ax1.plot(x1, y2, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
# ax1.plot(x1, y3, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
# ax1.plot(x1, y4, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
# ax1.plot(x1, y5, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
# ax1.set_xlabel('Cumulative Round of Communication')
# ax1.set_xticks([i for i in range(0, 41) if i % 10 == 0])
# ax1.set_ylabel('Test F1-Score')
# ax1.set_ylim(0, 1)  # 设置y轴的上下限
# ax1.grid(True)
# ax1.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')


ax2 = fig.add_subplot(gs1[0])
ax2.text(-0.3, 0.95, "GCN", transform=ax2.transAxes, fontdict={'weight': 'bold'})
ax2.plot(x1, y1_FedAvg_GCN, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y2_FedSem_FT_GCN, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y3_FedMatch_FT_GCN, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y4_FLwF_GCN, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y5_FedMH_GCN, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax2.set_xlabel('Cumulative Round of Communication')
ax2.set_xticks([i for i in range(0, 41) if i % 10 == 0])
ax2.set_ylabel('Test F1-Score')

ax2.grid(True)
ax2.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')


# ax3 = fig.add_subplot(gs1[1])
# ax3.text(-0.3, 0.95, "GAT", transform=ax3.transAxes, fontdict={'weight': 'bold'})
# ax3.plot(x1, y1_FedAvg_GAT, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
# ax3.plot(x1, y2_FedSem_FT_GAT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
# ax3.plot(x1, y3_FedMatch_FT_GAT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
# ax3.plot(x1, y4_FLwF_GAT, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
# ax3.plot(x1, y5_FedMH_GAT, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
# ax3.set_xlabel('Cumulative Round of Communication')
# ax3.set_xticks([i for i in range(0, 41) if i % 10 == 0])
# ax3.set_ylabel('Test F1-Score')
# ax3.grid(True)
# ax3.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')


# ax4 = fig.add_subplot(gs1[2])
# ax4.text(-0.5, 0.95, "GCN+LSTM", transform=ax4.transAxes, fontdict={'weight': 'bold'})
# ax4.plot(x1, y1_FedAvg_GCN_LSTM, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
# ax4.plot(x1, y2_FedSem_FT_GCN_LSTM, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
# ax4.plot(x1, y3_FedMatch_FT_GCN_LSTM, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
# ax4.plot(x1, y4_FLwF_GCN_LSTM, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
# ax4.plot(x1, y5_FedMH_GCN_LSTM, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
# ax4.set_xlabel('Cumulative Round of Communication')
# ax4.set_xticks([i for i in range(0, 41) if i % 10 == 0])
# ax4.set_ylabel('Test F1-Score')
# ax4.grid(True)
# ax4.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')



# ax5 = fig.add_subplot(gs2[0])
# ax5.set_title('Training Apprentice Model and Continual Update', fontweight='bold')
# ax5.plot(x2, y6, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
# ax5.plot(x2, y7, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
# ax5.plot(x2, y8, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
# ax5.plot(x2, y9, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
# ax5.plot(x2, y10, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
# ax5.set_xlabel('Cumulative Round of Communication', )
# ax5.set_xticks([i for i in range(40, 121) if i % 10 == 0])
# ax5.set_ylabel('Test F1-Score', )
# ax5.grid(True)
# ax5.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')


ax6 = fig.add_subplot(gs2[0])
ax6.plot(x2, y6_FedAvg_GCN, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y7_FedSem_FT_GCN, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y8_FedMatch_FT_GCN, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y9_FLwF_GCN, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y10_FedMH_GCN, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax6.set_xlabel('Cumulative Round of Communication', )
ax6.set_xticks([i for i in range(40, 121) if i % 10 == 0])
ax6.set_ylabel('Test F1-Score', )
ax6.grid(True)
ax6.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

# 后面两个图
# ax7 = fig.add_subplot(gs2[1])
# ax7.plot(x2, y6_FedAvg_GAT, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
# ax7.plot(x2, y7_FedSem_FT_GAT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
# ax7.plot(x2, y8_FedMatch_FT_GAT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
# ax7.plot(x2, y9_FLwF_GAT ,label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
# ax7.plot(x2, y10_FedMH_GAT, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
# ax7.set_xlabel('Cumulative Round of Communication', )
# ax7.set_xticks([i for i in range(40, 121) if i % 10 == 0])
# ax7.set_ylabel('Test F1-Score', )
# ax7.grid(True)
# ax7.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')


# ax8 = fig.add_subplot(gs2[2])
# ax8.plot(x2, y6_FedAvg_GCN_LSTM, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
# ax8.plot(x2, y7_FedSem_FT_GCN_LSTM, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
# ax8.plot(x2, y8_FedMatch_FT_GCN_LSTM, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
# ax8.plot(x2, y9_FLwF_GCN_LSTM, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
# ax8.plot(x2, y10_FedMH_GCN_LSTM, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
# ax8.set_xlabel('Cumulative Round of Communication', )
# ax8.set_xticks([i for i in range(40, 121) if i % 10 == 0])
# ax8.set_ylabel('Test F1-Score', )
# ax8.grid(True)
# ax8.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

ax2.set_ylim(0, 1)  # 设置y轴的上下限

ax6.set_ylim(0, 1)  # 设置y轴的上下限
# ax3.set_ylim(0, 1)  # 设置y轴的上下限
# ax4.set_ylim(0, 1)  # 设置y轴的上下限
# ax7.set_ylim(0, 1)  # 设置y轴的上下限
# ax8.set_ylim(0, 1)  # 设置y轴的上下限

plt.tight_layout()

plt.savefig(os.path.join(os.getcwd(), f"{filename}.png"))
plt.show()