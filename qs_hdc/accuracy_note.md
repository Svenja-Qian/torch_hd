# HDC Baseline Analysis Report

**Date:** 2026-01-12
**Environment:** macOS, Python 3.12.3, TorchHD 5.8.4

## 1. Experimental Settings

所有基准测试均在相同的硬件环境和超参数设置下运行，以确保公平对比。

### Common Parameters
*   **Hypervector Dimensions ($D$):** 4000
*   **Batch Size:** 1
*   **Device:** CPU
*   **Projection Encoding Method:** Random Projection (Binary, $+1/-1$ weights)
*   **Similarity Metric:** Hamming Similarity
*   **Learning Method:** Online Accumulation (Centroid-based)

### Dataset Specifics

| Dataset | Input Features | Classes | Data Type | Transform / Preprocessing |
| :--- | :--- | :--- | :--- | :--- |
| **ISOLET** | 617 | 26 | Audio (Speech) | None (Pre-extracted features) |
| **UCIHAR** | 561 | 6 | Sensor (IMU) | None (Pre-extracted features) |
| **MNIST** | 784 ($28 \times 28$) | 10 | Image | Flattened |
| **EMG** | 1024 ($256 \times 4$) | 5 | Bio-signal (Time-series) | Flattened (Cross-subject split) |
| **EuroLanguages** | 27 | 21 | Text | Character Histogram (Bag-of-Characters) |

---

## 2. Performance Comparison

| Baseline Script | Dataset | Accuracy (%) | Performance Level |
| :--- | :--- | :--- | :--- |
| `baseline_isolet.py` | **ISOLET** | **85.25%** | ⭐⭐⭐ High |
| `baseline_ucihar.py` | **UCIHAR** | **85.14%** | ⭐⭐⭐ High |
| `baseline_mnist.py` | **MNIST** | **80.21%** | ⭐⭐ Medium |
| `baseline_emg.py` | **EMG Hand Gestures** | **68.16%** | ⭐ Low-Medium |
| `baseline_EuroLanguages.py` | **European Languages** | **44.08%** | ❌ Low |

---

## 3. Analysis & Discussion

### 3.1 高精度表现分析 (ISOLET, UCIHAR)
**ISOLET (85.25%)** 和 **UCIHAR (85.14%)** 取得了最好的成绩。
*   **特征适配性强**: 这两个数据集提供的都是预提取的高质量特征（Spectral coefficients for Audio, Statistical features for IMU）。这些特征向量在原始空间中已经具有较好的线性可分性，经过随机投影到高维空间后，HDC 能够很好地保持这种距离关系（Johnson-Lindenstrauss lemma）。
*   **维度优势**: 输入特征维度适中 (~600)，投影到 4000 维时信息损失较小，且正交性保持较好。

### 3.2 中等精度表现分析 (MNIST)
**MNIST (80.21%)** 表现中规中矩。
*   **空间结构丢失**: 当前的基准实现直接将 $28 \times 28$ 的图像展平（Flatten）为 784 维向量进行投影。这种做法完全忽略了像素间的二维空间邻域关系。
*   **改进潜力**: 如果使用 ID-Level Encoding 或 Permutation 来编码像素位置信息，准确率通常可以提升到 90% 以上。目前的 80% 仅反映了像素值的统计分布特征。

### 3.3 低精度表现分析 (EMG, EuroLanguages)

#### **EMG Hand Gestures (68.16%)**
*   **跨受试者挑战 (Cross-subject Difficulty)**: 训练集包含受试者 0-3，测试集是受试者 4。肌电信号（EMG）在不同人之间差异巨大（由于肌肉解剖结构、皮肤阻抗、电极位置微小偏移等）。简单的随机投影无法提取出“受试者不变”的特征。
*   **时序信息利用不足**: 类似于 MNIST，代码将 $256 \times 4$ 的时间序列直接展平。虽然保留了全部数据，但没有利用 HDC 的 N-gram 或 Permutation 操作来显式编码时间动态（Temporal Dynamics），导致模型难以捕捉手势的动态变化模式。

#### **European Languages (44.08%)**
这是表现最差的一组，原因主要有以下几点：
1.  **特征过于简单 (Information Loss)**:
    *   代码使用了 `transform` 函数计算**字符直方图 (Character Histogram)**。这意味着输入只有 27 个数值（26个字母+空格的频率）。
    *   **丢失了语序信息**: 语言的关键特征在于字母和单词的组合顺序（如 "the", "ing" 等 N-grams）。仅靠字母频率（如 'e' 在英语和法语中都很高）很难区分 21 种欧洲语言，尤其是同语族语言之间（如西班牙语和葡萄牙语）。
2.  **低维输入投影**: 将仅 27 维的特征投影到 4000 维，虽然增加了维度，但原始信息量本身就非常有限（Information Bottleneck）。
3.  **改进方向**: 使用 N-gram 编码（如 Trigram hypervectors）并结合 Permutation 操作，通常能将语言识别任务的准确率提升到 90% 以上。

---

## 4. 总结

本次基准测试表明，**随机投影（Random Projection）** 是一种通用的特征编码方法，但其效果高度依赖于原始输入特征的质量和表达能力。
*   对于**预处理特征丰富**的数据（ISOLET, UCIHAR），简单 HDC 效果很好。
*   对于**结构化数据**（图像、时间序列、文本），简单的展平或统计直方图会丢失关键的空间/时间结构信息，导致准确率受限。需要在编码阶段引入更高级的 HDC 算子（如 Permutation, Binding）来提升性能。
