---
title: 手撕kmeans（简易版）
author: Hygen
description: 华为11.19机试题第21题
tags: [Solution, ML]
published: 2025-11-20
---
# 题目描述
简单来说就是给定一个输入`k, m, n`
- `k`表示聚类的数量
- `m`表示样本的数量
- `n`表示训练轮数

样本的特征维度固定是`4`

接下来`m`行输入每个样本的值，每行包含4个`浮点数`，每个浮点数表示一个特征

要求直接选择前`k`个样本作为初始化质心

训练结束条件为`训练次数达到n轮`或`新簇和旧簇的距离<1e-8`，后者意为如果训练前后两个簇位置没变，则说明已经稳定

输入保证所有数据都合法且已归一化

输出：

要输出`k`个值，表示这`k`个聚类中每个簇有几个样本，按`从小到大`的顺序排列
# 样例
## 输入
```
3 20 1000
0.11 0.79 0.68 0.97
1.0 0.8 0.13 0.33
0.27 0.02 0.5 0.46
0.83 0.29 0.23 0.75
0.97 0.08 0.84 0.55
0.29 0.71 0.17 0.83
0.03 0.6 0.88 0.28
0.24 0.26 0.82 0.03
0.96 0.12 0.82 0.36
0.13 0.12 0.86 0.44
0.23 0.7 0.35 0.06
0.42 0.49 0.67 0.84
0.8 0.49 0.47 0.7
0.68 0.03 0.11 0.07
0.77 0.19 0.95 0.44
0.25 0.12 0.98 0.04
0.7 0.11 0.53 0.3
0.73 0.67 0.46 0.96
0.11 0.31 0.91 0.57
0.43 0.61 0.13 0.1
```
表示将20个样本分为3个簇，训练1000轮
## 输出
```
4 6 10
```
表示分为的三个簇中第一个簇有4个样本，第二个簇有6个样本，第三个簇有10个样本

# 解题思路

## 核心思想

K-means 是一种经典的**无监督聚类算法**，其目标是将 $m$ 个样本划分为 $k$ 个簇，使得同一簇内的样本相似度高，不同簇之间的样本相似度低。

算法的核心思想是：
1. 初始化 $k$ 个聚类中心
2. 将每个样本分配到最近的聚类中心
3. 根据分配结果更新聚类中心（取簇内样本的均值）
4. 重复步骤 2-3，直到聚类中心收敛或达到最大迭代次数

## 算法步骤

### 1. 初始化聚类中心

根据题意，直接选择前 $k$ 个样本作为初始聚类中心。

```python
  def _init_centroids(self, data):
    self.centroids = data[: self.n_clusters]
```

### 2. 计算距离矩阵

对于每个样本，计算其到所有 $k$ 个聚类中心的**欧氏距离**。使用 NumPy 的广播机制高效计算，并将结果存储在 `self.distances` 中：

```python
  def _distance(self, data):
    self.distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
```

> 📖 **补充知识**：关于 `np.newaxis`、`np.sqrt`、`np.sum` 和 `axis=2` 的详细说明，请参考 [距离计算相关函数](#距离计算相关函数)

### 3. 分配样本到最近的簇

使用 `argmin` 找到每个样本距离最近的聚类中心索引。该方法直接使用 `self.distances` 中存储的距离矩阵：

```python
  def _min_distance(self):
    return np.argmin(self.distances, axis=1)
```

> 📖 **补充知识**：关于 `np.argmin` 和 `axis=1` 的详细说明，请参考 [argmin 函数详解](#argmin-函数详解)

### 4. 更新聚类中心

对于每个簇，计算簇内所有样本的**均值**作为新的聚类中心。使用列表推导式高效计算，并返回新旧聚类中心的欧氏范数（用于判断收敛）：

```python
  def _update_centroids(self, data):
    old = self.centroids
    self.centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
    return np.linalg.norm(old - self.centroids)
```

> 📖 **补充知识**：关于 `np.mean`、`axis=0` 和 `np.linalg.norm` 的详细说明，请参考 [更新聚类中心相关函数](#更新聚类中心相关函数) 和 [范数计算函数](#范数计算函数)

### 5. 迭代训练

重复执行分配和更新步骤，直到满足以下条件之一：
- 达到最大迭代次数 $n$
- 聚类中心的变化小于阈值 $10^{-8}$（使用欧氏范数衡量）

```python
  def fit(self, data):
    self._init_centroids(data)
    for _ in range(self.max_iter):
      self._distance(data)
      self.labels = self._min_distance()
      if self._update_centroids(data) < self.tolerance:
        break
```

> 📖 **补充知识**：关于 `np.linalg.norm` 的详细说明，请参考 [范数计算函数](#范数计算函数)

### 6. 统计并输出结果

训练完成后，统计每个簇的样本数量，并按从小到大排序输出：

```python
  cnt = np.bincount(kmeans.labels, minlength=k)
  cnt.sort()
  print(*cnt)
```

> 📖 **补充知识**：关于 `np.bincount`、`np.array` 和 `reshape` 的详细说明，请参考 [数据统计相关函数](#数据统计相关函数)

## 关键实现细节

1. **距离计算优化**：使用 NumPy 的广播机制一次性计算所有样本到所有聚类中心的距离，避免嵌套循环，提高效率。距离矩阵存储在 `self.distances` 中，供后续使用。

2. **代码结构优化**：
   - 使用私有方法（以下划线开头）封装内部实现细节
   - `_update_centroids` 方法使用列表推导式简化代码，并直接返回收敛度
   - `_min_distance` 方法直接使用 `self.distances`，无需传递参数

3. **收敛判断**：`_update_centroids` 方法直接返回新旧聚类中心的欧氏范数，在 `fit` 方法中判断是否小于阈值 $10^{-8}$ 来决定是否提前终止迭代。

## 时间复杂度

- **时间复杂度**：$O(m \cdot k \cdot d \cdot t)$，其中 $m$ 是样本数，$k$ 是聚类数，$d$ 是特征维度（本题为 4），$t$ 是迭代次数。每次迭代需要计算距离矩阵和更新聚类中心。
- **空间复杂度**：$O(m \cdot k + k \cdot d)$，用于存储距离矩阵、标签和聚类中心。

## 注意事项

- K-means 算法对初始聚类中心的选择敏感，不同的初始化可能导致不同的聚类结果。
- 算法保证收敛，但可能收敛到局部最优解而非全局最优解。
- 本题中特征维度固定为 4，简化了实现复杂度。

# 代码实现
```python
import numpy as np
import sys

class Kmeans:
  def __init__(self, n_clusters, max_iter):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.centroids = None
    self.labels = None
    self.distances = None
    self.tolerance = 1e-8

  def _init_centroids(self, data):
    self.centroids = data[: self.n_clusters]

  def _distance(self, data):
    self.distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))

  def _min_distance(self):
    return np.argmin(self.distances, axis=1)

  def _update_centroids(self, data):
    old = self.centroids
    self.centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
    return np.linalg.norm(old - self.centroids)

  def fit(self, data):
    self._init_centroids(data)
    for _ in range(self.max_iter):
      self._distance(data)
      self.labels = self._min_distance()
      if self._update_centroids(data) < self.tolerance:
        break


def main():
  data = sys.stdin.read().split()
  idx = 0
  k, m, n = [int(x) for x in data[idx:idx+3]]
  idx += 3
  X = np.array(data[idx:idx+4*m], dtype=np.float64).reshape(m, 4)
  kmeans = Kmeans(n_clusters=k, max_iter=n)
  kmeans.fit(X)
  cnt = np.bincount(kmeans.labels, minlength=k)
  cnt.sort()
  print(*cnt)

if __name__ == '__main__':
  main()
```

# 补充知识

## 距离计算相关函数

### `np.newaxis`

**定义**：`np.newaxis` 是 `None` 的别名，用于数组索引时增加维度。

**示例**：
```python
arr = np.array([1, 2, 3])  # shape: (3,)
arr_new = arr[:, np.newaxis]  # shape: (3, 1)
# 结果: [[1], [2], [3]]
```

### `np.sqrt`

**定义**：对数组中的每个元素计算平方根。

**示例**：
```python
arr = np.array([4, 9, 16])
result = np.sqrt(arr)  # [2.0, 3.0, 4.0]
```

### `np.sum`

**定义**：`np.sum(a, axis=None)` 计算数组元素的和，`axis` 指定沿哪个轴求和。

**示例**：
```python
arr = np.array([[1, 2], [3, 4]])  # shape: (2, 2)
np.sum(arr, axis=0)  # 沿第0轴（行）求和: [4, 6]
np.sum(arr, axis=1)  # 沿第1轴（列）求和: [3, 7]
np.sum(arr)  # 所有元素求和: 10
```

### 距离计算的详细过程

1. **形状变换**：
   - `X` 的形状为 $(m, 4)$，表示 $m$ 个样本，每个样本有 4 个特征
   - `X[:, np.newaxis]` 将形状变为 $(m, 1, 4)$，增加了一个维度
   - `self.centroids` 的形状为 $(k, 4)$，表示 $k$ 个聚类中心

2. **广播运算**：
   - `X[:, np.newaxis] - self.centroids` 通过广播机制，将 $(m, 1, 4)$ 和 $(k, 4)$ 进行运算
   - 广播规则：将 `self.centroids` 扩展为 $(1, k, 4)$，然后与 $(m, 1, 4)$ 运算
   - 结果形状为 $(m, k, 4)$，表示每个样本到每个聚类中心在每个特征维度上的差值

3. **平方、求和和开方**：
   - `** 2` 对差值进行平方，形状仍为 $(m, k, 4)$
   - `.sum(axis=2)` **沿第 2 轴（特征维度）求和**，将 4 个特征维度的平方差相加
   - 结果形状变为 $(m, k)$，表示每个样本到每个聚类中心的欧氏距离的平方
   - `np.sqrt(...)` 对整个求和结果开方，得到最终的欧氏距离矩阵
   - 注意：括号 `((...).sum(axis=2))` 确保了先对平方差求和，然后再对结果开方，这是正确的欧氏距离计算公式

**为什么使用 `axis=2`？**

- 数组形状为 $(m, k, 4)$，三个维度分别表示：样本索引、聚类中心索引、特征维度
- `axis=2` 表示沿特征维度（第 3 个维度，索引为 2）求和
- 这样可以将每个样本到每个聚类中心在 4 个特征上的平方差相加，得到总的平方距离
- 如果使用 `axis=0` 或 `axis=1`，会沿错误的方向求和，无法得到正确的距离

**运算顺序说明**：

代码 `np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))` 的执行顺序是：
1. 计算差值并平方：`(data[:, np.newaxis] - self.centroids) ** 2` → 形状 $(m, k, 4)$
2. 沿特征维度求和：`.sum(axis=2)` → 形状 $(m, k)$，得到平方距离
3. 对平方距离开方：`np.sqrt(...)` → 形状 $(m, k)$，得到欧氏距离

外层括号 `((...).sum(axis=2))` 确保了求和操作在开方之前完成，这符合欧氏距离的数学定义：$\sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}$，即先对各个维度的平方差求和，再对总和开方。

## argmin 函数详解

### `np.argmin`

**定义**：`np.argmin(a, axis=None)` 返回沿指定轴的最小值的索引。

**示例**：
```python
arr = np.array([[3, 1, 4], [2, 5, 1]])  # shape: (2, 3)
np.argmin(arr, axis=0)  # 沿第0轴找最小值索引: [1, 0, 1]
np.argmin(arr, axis=1)  # 沿第1轴找最小值索引: [1, 2]
np.argmin(arr)  # 全局最小值索引（扁平化后）: 1
```

### 为什么使用 `axis=1`？

- `dts` 的形状为 $(m, k)$，表示 $m$ 个样本到 $k$ 个聚类中心的距离
- `axis=1` 表示沿第 1 轴（聚类中心维度）查找最小值
- 对于每个样本（第 0 轴），我们在 $k$ 个聚类中心（第 1 轴）中找到距离最小的那个
- 结果是一个长度为 $m$ 的一维数组，每个元素表示对应样本最近的聚类中心索引

**示例说明**：
```python
# 假设 dts = [[2.5, 1.2, 3.1],  # 样本0到3个聚类中心的距离
#            [1.8, 2.3, 0.9],  # 样本1到3个聚类中心的距离
#            [3.2, 2.1, 1.5]]  # 样本2到3个聚类中心的距离
# np.argmin(dts, axis=1) 返回: [1, 2, 2]
# 表示：样本0最近的是聚类中心1，样本1最近的是聚类中心2，样本2最近的是聚类中心2
```

## 更新聚类中心相关函数

### `np.mean`

**定义**：`np.mean(a, axis=None)` 计算数组元素的平均值，`axis` 指定沿哪个轴计算。

**示例**：
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
np.mean(arr, axis=0)  # 沿第0轴（行）求均值: [2.5, 3.5, 4.5]
np.mean(arr, axis=1)  # 沿第1轴（列）求均值: [2.0, 5.0]
np.mean(arr)  # 所有元素的均值: 3.5
```

### 更新过程的详细说明

优化后的代码使用列表推导式简化了实现：

1. **提取每个簇的样本并计算均值**：
   - `data[self.labels == i]` 使用布尔索引提取属于第 $i$ 个簇的所有样本
   - `.mean(axis=0)` **沿第 0 轴（样本维度）求均值**
   - 对于每个簇，计算簇内所有样本在每个特征维度上的平均值
   - 结果形状为 $(4,)$，表示 4 个特征的平均值

2. **构建新的聚类中心数组**：
   - `np.array([...])` 将列表推导式的结果转换为数组
   - 最终 `self.centroids` 的形状为 $(k, 4)$，每行代表一个聚类中心

3. **计算收敛度**：
   - `old = self.centroids` 保存更新前的聚类中心
   - `np.linalg.norm(old - self.centroids)` 计算新旧聚类中心的欧氏范数
   - 返回范数值，用于在 `fit` 方法中判断是否收敛

**为什么使用 `axis=0`？**

- `samples` 的形状为 $(n_i, 4)$，包含 $n_i$ 个样本，每个样本有 4 个特征
- `axis=0` 表示沿样本维度（第 0 轴）进行聚合操作
- 对于每个特征维度（第 1 轴），我们对所有样本在该特征上的值求平均
- 这样可以得到一个 4 维的向量，表示该簇在所有特征上的平均位置
- 如果使用 `axis=1`，会沿特征维度求均值，得到每个样本的平均特征值，这是错误的

**示例说明**：
```python
# 假设第0个簇有3个样本
samples = np.array([[1.0, 2.0, 3.0, 4.0],  # 样本1
                    [1.5, 2.5, 3.5, 4.5],  # 样本2
                    [2.0, 3.0, 4.0, 5.0]]) # 样本3
# shape: (3, 4)

# np.mean(samples, axis=0) 对每列（特征）求均值
# 结果: [1.5, 2.5, 3.5, 4.5]  # 4个特征的平均值

# 如果使用 axis=1，会对每行（样本）求均值
# np.mean(samples, axis=1) 结果: [2.5, 3.0, 3.5]  # 每个样本的平均特征值（错误）
```

## 范数计算函数

### `np.linalg.norm`

**定义**：`np.linalg.norm(x, ord=None, axis=None)` 计算数组的范数，默认计算欧氏范数（L2 范数）。

**示例**：
```python
vec = np.array([3, 4])
np.linalg.norm(vec)  # 欧氏范数: 5.0 (sqrt(3^2 + 4^2))

matrix = np.array([[1, 2], [3, 4]])
np.linalg.norm(matrix)  # 矩阵的Frobenius范数: 5.477...
np.linalg.norm(matrix, axis=0)  # 沿第0轴计算: [3.162..., 4.472...]
```

**收敛判断说明**：
- `self.centroids - old` 计算新旧聚类中心的差值，形状为 $(k, 4)$
- `np.linalg.norm()` 计算所有差值的欧氏范数，得到一个标量值
- 如果这个值小于阈值 $10^{-8}$，说明所有聚类中心的变化都很小，算法已收敛

## 数据统计相关函数

### `np.bincount`

**定义**：`np.bincount(x, weights=None, minlength=0)` 返回一个数组，索引 `i` 处的值表示 `i` 在输入数组中出现的次数。

**示例**：
```python
labels = np.array([0, 1, 1, 0, 2, 1])
cnt = np.bincount(labels)  # [2, 3, 1] 表示0出现2次，1出现3次，2出现1次
cnt = np.bincount(labels, minlength=5)  # [2, 3, 1, 0, 0] 保证输出长度至少为5
```

**统计过程说明**：
- `kmeans.labels` 是一个长度为 $m$ 的一维数组，每个元素表示对应样本所属的簇索引（0 到 $k-1$）
- `np.bincount(kmeans.labels, minlength=k)` 统计每个簇索引出现的次数
- `minlength=k` 确保输出数组长度至少为 $k$，即使某些簇没有样本也会输出 0
- 结果 `cnt` 是一个长度为 $k$ 的数组，`cnt[i]` 表示第 $i$ 个簇的样本数量
- `cnt.sort()` 对数组进行原地排序，从小到大排列
- `print(*cnt)` 使用解包操作打印数组元素，用空格分隔

### `np.array`

**定义**：从列表或其他数组创建 NumPy 数组。

**示例**：
```python
data = ['1.0', '2.0', '3.0', '4.0']
arr = np.array(data, dtype=np.float64)  # [1.0, 2.0, 3.0, 4.0]
```

### `reshape`

**定义**：改变数组的形状而不改变数据。

**示例**：
```python
arr = np.array([1, 2, 3, 4, 5, 6])
arr.reshape(2, 3)  # [[1, 2, 3], [4, 5, 6]]
arr.reshape(3, 2)  # [[1, 2], [3, 4], [5, 6]]
```

**数据读取说明**：
- `np.array(data[idx:idx+4*m], dtype=np.float64)` 将字符串列表转换为 NumPy 数组
- 切片 `data[idx:idx+4*m]` 提取从索引 `idx` 开始的 $4m$ 个元素（Python 切片是左闭右开区间）
- 因为有 $m$ 个样本，每个样本有 4 个特征，所以总共需要 $4m$ 个浮点数
- `.reshape(m, 4)` 将一维数组重塑为 $(m, 4)$ 的二维数组，每行代表一个样本