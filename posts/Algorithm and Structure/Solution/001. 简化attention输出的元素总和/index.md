---
title: 简化attention输出的元素总和
description: 华为2025秋招AI岗笔试题
published: 2025-11-11
category: Algorithm and Structure
tags: [Solution]
author: Hygen
---
本题出自2025年秋招-华为-9.17AI岗笔试
# 题目描述
给定三个正整数 n、m、h（均小于 100），构造如下数据并计算结果。

**数据构造规则：**
- 输入特征矩阵 **X** 为 n×m 的全 1 矩阵。
- 三个权重矩阵 **W1**、**W2**、**W3** 均为 m×h 的"上三角全 1"矩阵（按行列索引在主对角线及其上方位置为1，其余为0；当 m≥h 时视为按行列索引的上三角扩展）。
- 令 **Q = X·W1**，**K = X·W2**，**V = X·W3**；计算 **S = (Q·K^T) / sqrt(h)**
- softmax 按行做"归一化"：对任意行向量 r，softmax(r) 的每个元素等于该元素除以本行所有元素之和。
- **Y = softmax(S)·V**

**输出要求：** 求矩阵 **Y** 所有元素的和，四舍五入到整数后输出

**时间限制：** C/C++ 1秒，其他语言 2秒  
**空间限制：** C/C++ 256M，其他语言 512M

**输入描述：**
- 一行，三个正整数 n m h（均小于 100，且均>0）

**输出描述：**
- 一行，一个整数：矩阵 Y 的元素和（四舍五入后）

**示例1**
```
输入例子：5 4 3
输出例子：30
例子说明：h≤m，单行和为 1+2+3=6；总和=n×6=5×6=30
```
# 解题思路
本题是一个模拟题，它有数学推理后$O(1)$的解法，但是在此不多做讨论，我们主要讨论$O(nm)$的解法。，也就是模拟的做法

其实题目已经将要做什么告诉我们了，我们只需要按照题目的要求一步一步来即可，直接看代码实现即可
# 代码实现
```python
import numpy as np
# 输入n, m, h
n, m, h = map(int, input().split())
# 构造X
X = np.ones((n, m))
# 构造W
W = np.triu(np.ones((m, h)))
Q = X @ W
K = X @ W
V = X @ W
S = (Q @ K.T) / np.sqrt(h)
row_sums = S.sum(axis=1, keepdims=True)
softmax_S = S / row_sums
Y = softmax_S @ V
print(int(round(np.sum(Y))))
```
语法知识补充：
- `np.ones((n, m))`：生成一个 n×m 的全 1 矩阵
- `np.triu(np.ones((m, h)))`：生成一个 m×h 的上三角全 1 矩阵
- `@`：矩阵乘法运算符
- `np.sqrt(h)`：计算 h 的平方根
- `np.sum(S, axis=1, keepdims=True)`：计算 S 按行求和，并保持维度
- `np.round(Y)`：四舍五入 Y 的每个元素