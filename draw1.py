import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

# 读取 CSV 文件中的数据
def read_csv_data(filename, column_name):
    data = pd.read_csv(filename)
    return data[column_name].values

# 读取数据
x1 = np.arange(0, 40, 1)
x2 = np.arange(40, 120, 1)

y1_FedAvg = read_csv_data('f1_scores_FedAvg.csv', 'FedAvg')[:40]
y2_FedSem_FT = read_csv_data('f1_scores_FedSem_FT.csv', 'FedSem_FT')[:40]
y3_FedMatch_FT = read_csv_data('f1_scores_FedMatch_FT.csv', 'FedMatch_FT')[:40]
y4_FLwF = read_csv_data('f1_scores_FLwF.csv', 'FLwF')[:40]
y5_FedMH = read_csv_data('f1_scores_FedMH.csv', 'FedMH')[:40]

y6_FedAvg = read_csv_data('f1_scores_FedAvg_incremental.csv', 'FedAvg_incremental')[40:]
y7_FedSem_FT = read_csv_data('f1_scores_FedSem_FT_incremental.csv', 'FedSem_FT_incremental')[40:]
y8_FedMatch_FT = read_csv_data('f1_scores_FedMatch_FT_incremental.csv', 'FedMatch_FT_incremental')[40:]
y9_FLwF = read_csv_data('f1_scores_FLwF_incremental.csv', 'FLwF_incremental')[40:]
y10_FedMH = read_csv_data('f1_scores_FedMH_incremental.csv', 'FedMH_incremental')[40:]

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

ax1 = fig.add_subplot(gs1[0])
ax1.set_title('Training Expert Model', fontweight='bold')
ax1.text(-0.3, 0.95, "MLP", transform=ax1.transAxes, fontdict={'weight': 'bold'})
ax1.plot(x1, y1_FedAvg, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax1.plot(x1, y2_FedSem_FT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax1.plot(x1, y3_FedMatch_FT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax1.plot(x1, y4_FLwF, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax1.plot(x1, y5_FedMH, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax1.set_xlabel('Cumulative Round of Communication')
ax1.set_xticks([i for i in range(0, 41) if i % 10 == 0])
ax1.set_ylabel('Test F1-Score')
ax1.grid(True)
ax1.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

ax2 = fig.add_subplot(gs1[1])
ax2.text(-0.3, 0.95, "GCN", transform=ax2.transAxes, fontdict={'weight': 'bold'})
ax2.plot(x1, y1_FedAvg, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y2_FedSem_FT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y3_FedMatch_FT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y4_FLwF, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax2.plot(x1, y5_FedMH, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax2.set_xlabel('Cumulative Round of Communication')
ax2.set_xticks([i for i in range(0, 41) if i % 10 == 0])
ax2.set_ylabel('Test F1-Score')
ax2.grid(True)
ax2.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

ax3 = fig.add_subplot(gs1[2])
ax3.text(-0.3, 0.95, "GAT", transform=ax3.transAxes, fontdict={'weight': 'bold'})
ax3.plot(x1, y1_FedAvg, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax3.plot(x1, y2_FedSem_FT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax3.plot(x1, y3_FedMatch_FT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax3.plot(x1, y4_FLwF, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax3.plot(x1, y5_FedMH, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax3.set_xlabel('Cumulative Round of Communication')
ax3.set_xticks([i for i in range(0, 41) if i % 10 == 0])
ax3.set_ylabel('Test F1-Score')
ax3.grid(True)
ax3.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

ax4 = fig.add_subplot(gs1[3])
ax4.text(-0.5, 0.95, "GCN+LSTM", transform=ax4.transAxes, fontdict={'weight': 'bold'})
ax4.plot(x1, y1_FedAvg, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax4.plot(x1, y2_FedSem_FT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax4.plot(x1, y3_FedMatch_FT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax4.plot(x1, y4_FLwF, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax4.plot(x1, y5_FedMH, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax4.set_xlabel('Cumulative Round of Communication')
ax4.set_xticks([i for i in range(0, 41) if i % 10 == 0])
ax4.set_ylabel('Test F1-Score')
ax4.grid(True)
ax4.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

ax5 = fig.add_subplot(gs2[0])
ax5.set_title('Training Apprentice Model and Continual Update', fontweight='bold')
ax5.plot(x2, y6_FedAvg, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax5.plot(x2, y7_FedSem_FT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax5.plot(x2, y8_FedMatch_FT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax5.plot(x2, y9_FLwF, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax5.plot(x2, y10_FedMH, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax5.set_xlabel('Cumulative Round of Communication', )
ax5.set_xticks([i for i in range(40, 121) if i % 10 == 0])
ax5.set_ylabel('Test F1-Score', )
ax5.grid(True)
ax5.legend(['FedAvg', 'FedSem FT', 'FedMatch FT', 'FLwF', 'FedMH (ours)'], loc='lower right')

ax6 = fig.add_subplot(gs2[1])
ax6.plot(x2, y6_FedAvg, label='FedAvg', linestyle = style['FedAvg']['line'], color=style['FedAvg']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y7_FedSem_FT, label='FedSem_FT', linestyle = style['FedSem_FT']['line'], color=style['FedSem_FT']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y8_FedMatch_FT, label='FedMatch_FT', linestyle = style['FedMatch_FT']['line'], color=style['FedMatch_FT']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y9_FLwF, label='FLwF', linestyle = style['FLwF']['line'], color=style['FLwF']['color'], linewidth=style['linewidth'])
ax6.plot(x2, y10_FedMH, label='FedMH (ours)', linestyle = style['FedMH (ours)']['line'], color=style['FedMH (ours)']['color'], linewidth=style['linewidth'])
ax6.set_xlabel('Cumulative Round of Communication', )
ax6.set_xticks([i for i in range(40, 121) if i % 10 == 0])
ax6.set_ylabel('Test F1-Score', )
ax6.grid(True)

plt.show()