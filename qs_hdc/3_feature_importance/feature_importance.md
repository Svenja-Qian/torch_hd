Algorithm: Feature-Importance-Guided HDC Ensemble on ISOLET

Input:
    ISOLET train set D_train, test set D_test
    RF feature importance CSV: F (features 0..616 sorted by importance)
    Total HDC dimension: D_total
    Number of experts: E
    Validation ratio: val_ratio
    Weighting mode: weighting ∈ { "uniform", "val_acc", "boosting" }
    Common feature ratio: common_ratio
    Per-expert unique feature ratio: per_expert_ratio
    Number of runs: R
Output:
    Per-run baseline and ensemble accuracies
    Aggregated mean/std metrics and hardware bit estimates

1:  # ---------- Feature partition from importance ----------
2:  Let n_features ← 617
3:  Read CSV, sort features by (rank_perm, rank_avg) ascending to obtain index list F_sorted[1..n_features]
4:  common_count ← max(1, floor(n_features * common_ratio))
5:  unique_per_expert ← max(1, floor(n_features * per_expert_ratio))
6:  common_indices ← F_sorted[1..common_count]
7:  remaining ← F_sorted[common_count+1 .. n_features]
8:  total_unique_needed ← min(E * unique_per_expert, length(remaining))
9:  unique_pool ← first total_unique_needed indices of remaining
10: Initialize E empty lists expert_unique[1..E]
11: For pos = 0 .. total_unique_needed-1 do
12:     e ← (pos mod E) + 1
13:     Append unique_pool[pos] to expert_unique[e]
14: For e = 1 .. E do
15:     partitions[e] ← unique( common_indices ∪ expert_unique[e] )
16: baseline_k_global ← length(partitions[1])
17: sorted_all ← F_sorted   # global importance order
18:
19: # ---------- Dimension allocation ----------
20: base_dim ← floor(D_total / E)
21: For e = 1 .. E do
22:     dims[e] ← base_dim
23: residue ← D_total - base_dim * E
24: For i = 1 .. residue do
25:     dims[i] ← dims[i] + 1
26:
27: # ---------- Multiple runs with different seeds ----------
28: For r = 1 .. R do
29:     seed_base ← run_time + (r - 1)
30:     # Split train into train/val
31:     Randomly shuffle D_train with seed_base
32:     Split into train_subset and val_subset with ratio val_ratio
33:     Construct data loaders:
34:         L_train (on train_subset, batch=1, shuffled)
35:         L_val   (on val_subset,   batch=1)
36:         L_test  (on D_test,       batch=1)
37:
38:     # ---------- Baseline model with global top-K features ----------
39:     Sample seeds [baseline_seed, expert_seeds[1..E]] from a RNG seeded by seed_base
40:     baseline_k ← baseline_k_global
41:     baseline_idx ← first baseline_k indices of sorted_all
42:     Initialize baseline HDC classifier C_base with:
43:         input features = baseline_idx, dimensions = D_total
44:     Train C_base on L_train:
45:         For each sample (x, y) in L_train:
46:             x' ← select features x[baseline_idx], center x' by subtracting 0.5
47:             h ← sign( RandomProjection(x') ) ∈ {−1,+1}^D_total
48:             Accumulate h to the class prototype of label y
49:         After all samples, for each class c:
50:             centroid_c ← sign( accumulated_vector_c )
51:
52:     # ---------- Train ensemble experts ----------
53:     If weighting == "boosting" then
54:         # AdaBoost-like training on train_subset
55:         N ← |train_subset|
56:         Initialize sample weights w_i ← 1 / N,  i = 1..N
57:         For e = 1 .. E do
58:             Use w_i as sampling probabilities to build a weighted training set L_train_w
59:             Initialize HDC expert C_e with:
60:                 features = partitions[e], dimensions = dims[e]
61:             Train C_e on L_train_w with the same centroid-learning rule as C_base
62:             Evaluate C_e on full train_subset (ordered) to obtain predictions ŷ_i
63:             incorrect_i ← 1 if ŷ_i ≠ y_i else 0
64:             ε_e ← Σ_i w_i * incorrect_i    # weighted training error
65:             If ε_e ≤ 0 then
66:                 α_e ← large_constant        # expert almost perfect
67:             Else if ε_e ≥ 0.5 then
68:                 α_e ← small_constant        # weak expert
69:             Else
70:                 α_e ← 0.5 * log( (1 − ε_e) / max(ε_e, 1e−8) )
71:             Update sample weights:
72:                 w_i ← w_i * exp(α_e * incorrect_i), then normalize Σ_i w_i = 1
73:         Set ensemble weights w_e ← α_e   for e = 1..E
74:     Else
75:         # Independent training on train_subset
76:         For e = 1 .. E do
77:             Initialize HDC expert C_e with:
78:                 features = partitions[e], dimensions = dims[e]
79:             Train C_e on L_train (uniform sampling) using same centroid rule
80:         If weighting == "val_acc" then
81:             For e = 1 .. E do
82:                 acc_val[e] ← accuracy of C_e on L_val
83:             w_e ← acc_val[e] / (Σ_j acc_val[j])   # normalize by sum
84:         Else
85:             w_e ← 1 / E for all e                 # uniform weights
86:
87:     # ---------- Evaluation on test set ----------
88:     For each sample (x, y) in L_test do
89:         # Baseline prediction
90:         x_base ← x[baseline_idx], center and encode to h_base
91:         s_base[c] ← HammingSimilarity(h_base, centroid_c) for all classes c
92:         ŷ_base ← argmax_c s_base[c]
93:
94:         # Ensemble prediction (soft voting)
95:         For e = 1 .. E do
96:             x_e ← x[partitions[e]], center and encode to h_e
97:             s_e[c] ← HammingSimilarity(h_e, centroid_c^(e)) for all classes c
98:         S[c] ← Σ_e w_e * s_e[c]
99:         ŷ_ens ← argmax_c S[c]
100:        Update counters for baseline and ensemble accuracy
101:
102:    Record per-run accuracies and other metrics (e.g., hardware bits from dims and partitions)
103:
104: # ---------- After all runs ----------
105: Aggregate per-run metrics across r = 1..R:
106:     Compute mean and standard deviation of baseline and ensemble accuracies
107:     Optionally compute per-expert mean accuracies
108: Write a CSV row with:
109:     (D_total, E, dims, common_ratio, per_expert_ratio,
110:      importance file name, baseline_k_global, seeds, aggregated metrics, hardware bits, R)


------
主要思路概述

这个脚本是在 ISOLET 数据集上做基于特征重要性的 HDC（Hyperdimensional Computing）集成实验。整体流程可以概括为：

- 用随机森林得到的特征重要性排序，把 617 个特征拆成多个专家（experts）的子集：所有专家共享一部分“公共特征”，每个专家再有自己的一部分“独有特征”。
- 每个专家是一个二值 HDC 分类器：输入特征通过随机二值投影得到高维二值超向量，再按类做累加得到每一类的“超向量质心”，预测时用汉明相似度。
- 在给定总维度 D_total 和专家个数 E 的约束下，把总维度均分给各个专家（可能有少量余数分配给前几个专家），构建 E 个子维度的 HDC 专家。
- 训练和评估两条线：
  - 基线模型（baseline）：用“全局最重要的前 K 个特征”训练一个单一 HDC 模型，维度为 D_total。
  - 集成模型（ensemble）：用特征分牌构造的 E 个专家，按照不同的权重策略（均匀 / 按验证集精度 / boosting）做加权 soft voting。
- 多次重复实验（不同随机种子、数据划分和投影），记录每次的基线/集成精度，最后汇总均值和标准差，并写入 CSV，辅助画图和论文分析。

核心组件说明
- Classifier ：单个 HDC 分类器
  
  - 输入：类别数 num_classes ，维度 dimensions ，输入特征数 in_features ，运行设备 device ，以及可选的特征索引 feature_indices 。
  - 内部：
    - 一个 torchhd.embeddings.Projection(in_features, dimensions) 的投影矩阵，并强制权重为 ±1 的二值矩阵。
    - encode(x) ：
      - 如果提供 feature_indices ，先做特征子集选择；
      - 将特征平移（ x - 0.5 ）；
      - 通过投影矩阵映射到高维空间；
      - 阈值化成二值超向量（布尔/±1）。
    - fit(data_loader) ：
      - 初始化每个类别的累积向量；
      - 对训练数据逐样本编码成 ±1 超向量，并根据标签把该向量加到对应类别的累积中；
      - 对每个维度取多数符号，得到每个类别的类质心（class centroid），也是二值超向量。
    - forward(samples) ：
      - 对输入样本编码成超向量；
      - 计算与各类别质心的汉明相似度；
      - 返回对每个类别的相似度分数。
    - predict(samples) ：返回相似度最大对应的类别。
- 特征重要性分牌：
  
  - load_feature_importance_indices(...) ：
    - 从 CSV 中读入随机森林的特征重要性（字段包括 feature , rank_perm , rank_avg 等）。
    - 按 rank_perm 、 rank_avg 升序排序，得到全局特征排序列表。
    - 将特征名映射为 [0, 616] 的特征索引。
    - 设：
      - common_ratio ：公共特征占比；
      - per_expert_ratio ：每个专家独有特征占比；
      - E ：专家数量。
    - 从排序列表中：
      - 前 common_ratio * n_features 个特征作为所有专家共享的公共特征。
      - 之后的一部分特征作为独有特征池，按轮转（round-robin）分配给各个专家，每个专家获得约 per_expert_ratio * n_features 个独有特征。
    - 对每个专家 e，生成一个特征索引集合
       partition[e] = unique(common_indices ∪ expert_unique[e]) 。
  - parse_sorted_feature_indices(...) ：仅按重要性返回前 n_features 个排序索引，用于构建基线的“全局 top-K 特征”。
- 专家训练与集成：
  
  - train_expert(idx_e, D_e, seed, train_loader, device) ：
    - 固定随机种子；
    - 用特征子集 idx_e 构造一个维度为 D_e 的 Classifier ；
    - 在给定的 train_loader 上完成 fit() 。
  - evaluate_accuracy(model, data_loader) ：在给定数据上计算精度。
  - soft_vote_weighted(experts, samples, weights) ：
    - 对每个专家 i，计算其对样本的相似度向量 sim_i ；
    - 按权重 weights[i] 加权相加得到总相似度；
    - 对每个样本取 argmax 得到最终类别，形成加权 soft voting 集成。
- Boosting 风格训练：
  
  - train_experts_boosting(E, partitions, dims, train_subset, train_eval_loader, device, expert_seeds) ：
    - 在训练子集上初始化样本权重 w_i = 1/N 。
    - 对 e=1..E：
      - 用 WeightedRandomSampler 按当前样本权重有放回采样，构造训练 loader；
      - 用该加权训练集训练专家 e（特征子集为 partitions[e] ，维度为 dims[e] ）；
      - 在 train_eval_loader 上评估该专家，得到预测 preds ；
         计算加权错误率
         ε_e = Σ_i w_i · [pred_i != label_i] ；
      - 根据错误率确定权重系数：
        - 错误率很低时给一个较大的固定 alpha；
        - 错误率 ≥ 0.5 时给一个很小的 alpha；
        - 否则用 AdaBoost 形式
           α_e = 0.5 * log( (1 - ε_e) / ε_e ) ；
      - 更新样本权重：
         w_i ← w_i · exp(α_e · [pred_i != label_i]) ，再归一化。
    - 返回训练好的专家列表和对应的 α 列表。后续在集成时用 α 作为专家权重。
    