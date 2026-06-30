---
title: 找到最高海拔
author: Hygen
description: LeetCode-1732
category: Algorithm and Structure
tags: [Solution, 前缀和]
published: 2026-06-19
---
# 题目描述
[题目链接](https://leetcode.cn/problems/find-the-highest-altitude/description/?envType=daily-question&envId=2026-06-19)

有一个自行车手打算进行一场公路骑行，这条路线总共由 `n + 1` 个不同海拔的点组成。自行车手从海拔为 `0` 的点 `0` 开始骑行。

给你一个长度为 `n` 的整数数组 `gain`，其中 `gain[i]` 是点 `i` 和点 `i + 1` 的**净海拔增益**（可能为负）。请你返回自行车手在骑行过程中达到的**最高海拔**。

## 输入
整数数组 `gain`
## 输出
骑行过程中达到的最高海拔
## 约束条件
1. `1 <= gain.length <= 100`
2. `-100 <= gain[i] <= 100`
## 示例1
```
input: gain = [-5,1,5,0,-7]
output: 1
```
解释：海拔依次为 `0, -5, -4, 1, 1, -6`，最高海拔为 `1`。
## 示例2
```
input: gain = [-4,-3,-2,-1,4,3,2]
output: 0
```
解释：全程海拔始终不高于起点 `0`。
# 解题思路
题目给出了相邻两点之间的海拔**变化量**，起点海拔固定为 `0`。因此，每经过一个路段，当前海拔就是在原有基础上加上对应的 `gain[i]`——本质上是在做**前缀和**。

## 从逐段累加出发
最自然的想法是：从海拔 `0` 出发，依次走过每一段路，记录途中出现过的最大海拔。

以 `gain = [-5, 1, 5, 0, -7]` 为例，逐段累加后各点的海拔为：

| 路段 | `gain[i]` | 当前海拔 |
| --- | --- | --- |
| 起点 | — | 0 |
| 第 1 段 | -5 | -5 |
| 第 2 段 | 1 | -4 |
| 第 3 段 | 5 | 1 |
| 第 4 段 | 0 | 1 |
| 第 5 段 | -7 | -6 |

最高海拔出现在第 3 段之后，为 `1`。

## 前缀和：用 `accumulate` 一次求出各点海拔
「逐段累加」若把每一步的结果都记下来，就得到了 `gain` 的**前缀和**序列——第 `i` 项表示走完前 `i` 段路后的海拔。

Python 标准库 `itertools.accumulate` 正好做这件事：它按顺序把 `gain` 累加，生成每一步的当前海拔。题目要求的是这些海拔（连同起点 `0`）中的最大值，因此：

$$
\text{ans} = \max\bigl(0,\ \text{accumulate}(\text{gain})\bigr)
$$

外层再与 `0` 取最大，是因为起点海拔 `0` 本身也可能就是答案（如示例 2 全程不升反降）。

## 复杂度
- 时间复杂度：$O(n)$，遍历 `gain` 一次
- 空间复杂度：$O(1)$，`accumulate` 返回迭代器，`max` 边读边比较，不额外存储整个前缀和数组

## 代码
```python
from itertools import accumulate

class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        return max(max(accumulate(gain)), 0)
```

# 解题思路（单次遍历）
前缀和视角需要先「算出每一步的海拔」，再取最大。换一个角度：我们其实**不必保存**所有中间海拔，只需在遍历过程中维护两个量——当前海拔 `alt` 和历史最大 `ans`。

每读入一个 `gain[i]`，更新 `alt += gain[i]`，再用 `ans = max(ans, alt)` 刷新答案。起点 `0` 可作为 `ans` 的初始值，逻辑与上一节完全等价，只是把「先累加、再取 max」合并成了一次扫描。

以 `gain = [-5, 1, 5, 0, -7]` 为例：

| `gain[i]` | `alt` | `ans` |
| --- | --- | --- |
| — | 0 | 0 |
| -5 | -5 | 0 |
| 1 | -4 | 0 |
| 5 | 1 | 1 |
| 0 | 1 | 1 |
| -7 | -6 | 1 |

最终 `ans = 1`。时间、空间复杂度仍为 $O(n)$ 和 $O(1)$，适合在不方便使用 `itertools` 的环境（如部分 OJ）中手写。

# 代码优化
本题时间复杂度已是 $O(n)$，不存在渐近意义上的更优算法。优化空间主要在表达简洁度与可读性。

## 用解包替代嵌套 `max`
`max(max(accumulate(gain)), 0)` 需要读完整个迭代器才能得出内层最大值，再与 `0` 比较。利用 `max` 支持多参数的特性，把起点 `0` 一并传入更直观：

```python
return max(0, *accumulate(gain))
```

语义不变：在起点 `0` 与所有前缀和之间取最大。

## 显式循环（无需 `itertools`）
若更偏好「一眼能看懂在干什么」的写法，单次遍历同样紧凑：

```python
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        alt = ans = 0
        for g in gain:
            alt += g
            ans = max(ans, alt)
        return ans
```

与 `accumulate` 版本相比，少了标准库依赖，循环体也直接对应「逐段骑行、随时更新最高海拔」的题意——可按场景与个人习惯选择。
