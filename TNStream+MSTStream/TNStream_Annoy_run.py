"""
TNStream Annoy版本 - 运行脚本
使用Annoy索引的轻量高速版本
修改说明：导入TNStream类而不是TNStreamAnnoy
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from TNStream_Annoy import TNStream  # ✓ 修改：导入TNStream而不是TNStreamAnnoy
import time
import json
from datetime import datetime
import os

# 加载数据集
print("="*70)
print("TNStream Annoy版本 - 性能测试")
print("="*70)

dataset = np.loadtxt("/Users/tsiphenzeng/Desktop/TNStream/dataset/dataset2/kdd_converted.txt", dtype=float, delimiter=' ')
X = dataset[:, :-1]
labels_true = dataset[:, -1]

print(f"\n数据集大小: {X.shape[0]} 样本, {X.shape[1]} 维")
print(f"真实类别数: {len(np.unique(labels_true))}")

# 标准化只做一次
print("\n数据标准化...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("✓ 标准化完成")

# 参数设置
dataset_name = "KDD"
W = 4288
N = 3
r = 0.701
n_micro = 8
k = 4
mk = 4

print(f"\n参数配置:")
print(f"  W (窗口大小) = {W}")
print(f"  N (微簇最小点数) = {N}")
print(f"  r (微簇半径) = {r}")
print(f"  n_micro (宏簇最小微簇数) = {n_micro}")
print(f"  k (k-NN参数) = {k}")
print(f"  mk (共享近邻数) = {mk}")

# 结果存储
results = []
best_ari = -1
best_nmi = -1
best_purity = 0
best_result = None

# 创建日志文件
log_file = f"tnstream_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
results_file = f"tnstream_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
best_file = f"tnstream_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def log_message(msg):
    """记录到文件和控制台"""
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

log_message("="*70)
log_message(f"开始时间: {datetime.now()}")
log_message("="*70)

print(f"\n{'='*70}")
print("开始运行 TNStream Annoy版本")
print(f"{'='*70}\n")

T = 0
while T < 1000:
    T += 1

    print(f"[迭代 {T}/1000]", end=" ", flush=True)

    start = time.time()

    try:
        # ✓ 修改：使用TNStream类
        model = TNStream(
            X_scaled.copy(),
            labels_true.copy(),
            N=N,
            W=W,
            r=r,
            n_micro=n_micro,
            d=X.shape[1],
            plotFigure=0,
            k=k,
            mk=mk
        )

        elapsed = time.time() - start

        # 超时检查
        if elapsed > 60 * 20:
            print(f"超时 ({elapsed:.1f}s), 跳过")
            continue

        # 评估结果
        metrics_dict = model.evaluate()
        ARI = metrics_dict['ARI']
        Purity = metrics_dict['Purity']
        NMI = metrics_dict['NMI']

        print(f"用时: {elapsed:.2f}s | Purity: {Purity:.4f} | ARI: {ARI:.4f} | NMI: {NMI:.4f}", flush=True)

        # 保存结果
        result = {
            'iteration': T,
            'dataset': dataset_name,
            'params': {
                'W': W, 'N': N, 'r': r, 'n_micro': n_micro, 'k': k, 'mk': mk
            },
            'time': elapsed,
            'purity': Purity,
            'ari': ARI,
            'nmi': NMI,
            'num_mcs': len(model.MCs),
            'num_macro_cs': len(model.MacroClusters),
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)

        # 更新最优结果（多个指标综合考虑）
        combined_score = 0.4 * ARI + 0.3 * NMI + 0.3 * Purity
        best_combined = 0.4 * best_ari + 0.3 * best_nmi + 0.3 * best_purity if best_ari >= 0 else -1

        if combined_score > best_combined:
            best_ari = ARI
            best_nmi = NMI
            best_purity = Purity
            best_result = result
            print(f"  ⭐ 新的最优结果！(ARI: {ARI:.4f})")

        # 达到优秀水平时打印详细信息
        if ARI > 0.5 and NMI > 0.4:
            detail_msg = f"\n{'='*70}\n🏆 优秀结果 (迭代 {T})\n{'='*70}\n"
            detail_msg += f"数据集: {dataset_name}\n"
            detail_msg += f"参数: N={N}, r={r:.3f}, mk={mk}, n_micro={n_micro}\n"
            detail_msg += f"\n【性能指标】\n"
            detail_msg += f"  运行时间: {elapsed:.2f}s\n"
            detail_msg += f"  微簇数: {len(model.MCs)}\n"
            detail_msg += f"  宏簇数: {len(model.MacroClusters)}\n"
            detail_msg += f"\n【聚类质量】\n"
            detail_msg += f"  Purity = {Purity:.4f}\n"
            detail_msg += f"  ARI = {ARI:.4f}\n"
            detail_msg += f"  NMI = {NMI:.4f}\n"
            detail_msg += f"{'='*70}\n"
            log_message(detail_msg)

        # 每10次保存一次所有结果
        if T % 10 == 0:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"错误: {str(e)[:100]}")
        continue

# 最终总结
print(f"\n{'='*70}")
print("运行完成！")
print(f"{'='*70}\n")

if best_result:
    summary = f"""
【最优结果总结】
  迭代: {best_result['iteration']}
  Purity: {best_result['purity']:.6f}
  ARI: {best_result['ari']:.6f}
  NMI: {best_result['nmi']:.6f}
  运行时间: {best_result['time']:.2f}s
  微簇数: {best_result['num_mcs']}
  宏簇数: {best_result['num_macro_cs']}
  
  参数:
    N = {best_result['params']['N']}
    r = {best_result['params']['r']}
    mk = {best_result['params']['mk']}
    n_micro = {best_result['params']['n_micro']}
    k = {best_result['params']['k']}
"""
    print(summary)
    log_message(summary)

    # 保存最终结果
    with open(best_file, 'w', encoding='utf-8') as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)

    print(f"✓ 最优结果已保存到: {best_file}")

# 保存所有结果
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ 所有结果已保存到: {results_file}")
print(f"✓ 运行日志已保存到: {log_file}")

# 打印TOP 5结果
if results:
    print(f"\n{'='*70}")
    print("TOP 5 (按ARI排序)")
    print(f"{'='*70}")
    top_results = sorted(results, key=lambda x: x['ari'], reverse=True)[:5]
    for i, r in enumerate(top_results, 1):
        print(f"{i}. ARI={r['ari']:.4f}, NMI={r['nmi']:.4f}, Purity={r['purity']:.4f}, 时间={r['time']:.2f}s")

print(f"\n总共完成 {len(results)} 次有效运行\n")