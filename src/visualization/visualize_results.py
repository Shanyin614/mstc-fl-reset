"""
MSTC-FL 实验结果可视化脚本
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体（可选，如果需要中文标题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_results(results_path: str):
    """加载实验结果"""
    with open(results_path, 'rb') as f:
        history = pickle.load(f)
    return history


def plot_performance(history, save_dir):
    """绘制性能指标曲线"""
    rounds = range(len(history['round_stats']))
    acc = [r['global_acc'] for r in history['round_stats']]
    f1 = [r['global_f1'] for r in history['round_stats']]
    precision = [r['global_precision'] for r in history['round_stats']]
    recall = [r['global_recall'] for r in history['round_stats']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：准确率和 F1
    axes[0].plot(rounds, acc, 'o-', label='Accuracy', linewidth=2, markersize=6)
    axes[0].plot(rounds, f1, 's-', label='F1 Score', linewidth=2, markersize=6)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('MSTC-FL Global Performance', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.75, 0.90])

    # 右图：精确率和召回率
    axes[1].plot(rounds, precision, '^-', label='Precision', linewidth=2, markersize=6)
    axes[1].plot(rounds, recall, 'v-', label='Recall', linewidth=2, markersize=6)
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.70, 1.00])

    plt.tight_layout()
    plt.savefig(save_dir / 'performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir / 'performance.png'}")
    plt.close()


def plot_drift_analysis(history, save_dir):
    """绘制漂移检测分析"""
    rounds = range(len(history['round_stats']))
    n_drifts = [r['n_drifts'] for r in history['round_stats']]
    n_clusters = [r['n_clusters'] for r in history['round_stats']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：漂移检测
    axes[0].bar(rounds, n_drifts, color='coral', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Number of Drifted Clients', fontsize=12)
    axes[0].set_title('Concept Drift Detection (ADWIN++)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # 右图：聚类数量
    axes[1].plot(rounds, n_clusters, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Number of Active Clusters', fontsize=12)
    axes[1].set_title('Hierarchical GCN Clustering', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 10])

    plt.tight_layout()
    plt.savefig(save_dir / 'drift_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir / 'drift_analysis.png'}")
    plt.close()


def plot_ensemble_history(history, save_dir):
    """绘制集成成员性能历史"""
    em_history = history['em_history']

    if len(em_history) == 0:
        print("⚠️  No ensemble history to plot")
        return

    em_history_array = np.array(em_history)
    n_rounds, n_ensemble = em_history_array.shape

    plt.figure(figsize=(12, 6))
    for i in range(n_ensemble):
        plt.plot(range(n_rounds), em_history_array[:, i],
                 marker='o', label=f'EM {i}', linewidth=2, markersize=4)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ensemble Members Performance History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'ensemble_history.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir / 'ensemble_history.png'}")
    plt.close()


def print_summary(history):
    """打印实验摘要"""
    round_stats = history['round_stats']

    print("\n" + "=" * 70)
    print("MSTC-FL Experiment Summary")
    print("=" * 70)
    print(f"Total Rounds:        {len(round_stats)}")
    print(f"Best Accuracy:       {max([r['global_acc'] for r in round_stats]):.4f}")
    print(f"Best F1 Score:       {max([r['global_f1'] for r in round_stats]):.4f}")
    print(f"Final Accuracy:      {round_stats[-1]['global_acc']:.4f}")
    print(f"Final F1 Score:      {round_stats[-1]['global_f1']:.4f}")
    print(f"Total Drifts:        {sum([r['n_drifts'] for r in round_stats])}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 路径设置
    ROOT = Path(__file__).resolve().parents[1]
    results_path = ROOT / "results" / "mstc_fl_history.pkl"
    save_dir = ROOT / "results" / "figures"
    save_dir.mkdir(exist_ok=True)

    print("🎨 MSTC-FL Results Visualization\n")

    # 加载结果
    print("📂 Loading results...")
    history = load_results(results_path)

    # 打印摘要
    print_summary(history)

    # 绘图
    print("🖼️  Generating plots...")
    plot_performance(history, save_dir)
    plot_drift_analysis(history, save_dir)
    plot_ensemble_history(history, save_dir)

    print(f"\n✅ All plots saved to {save_dir}/")
    print("🎉 Visualization completed!")
