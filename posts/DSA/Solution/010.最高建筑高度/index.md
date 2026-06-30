---
title: 最高建筑高度
author: Hygen
description: LeetCode-1840
category: Algorithm and Structure
tags: [Solution, 排序, 贪心]
published: 2026-06-20
---
# 题目描述
[题目链接](https://leetcode.cn/problems/maximum-building-height/?envType=daily-question&envId=2026-06-20)

在一座城市里，你需要建 `n` 栋新的建筑，编号从 `1` 到 `n` 排成一列。城市有如下规定：

- 每栋建筑的高度必须是非负整数；
- 第 `1` 栋建筑的高度**必须**是 `0`；
- 任意两栋相邻建筑的高度差**不能超过** `1`。

此外，部分建筑还有额外的高度上限。`restrictions[i] = [idi, maxHeighti]` 表示建筑 `idi` 的高度不能超过 `maxHeighti`。题目保证每个 `idi` 唯一，且建筑 `1` 不会出现在 `restrictions` 中。

请你返回所有建筑中**最高**的那栋能达到的**最大高度**。

## 输入
整数 `n`，二维整数数组 `restrictions`
## 输出
最高建筑能达到的最大高度
## 约束条件
1. `2 <= n <= 10^9`
2. `0 <= restrictions.length <= min(n - 1, 10^5)`
3. `2 <= idi <= n`，`idi` 互不相同
4. `0 <= maxHeighti <= 10^9`
## 示例1
```
input: n = 5, restrictions = [[2,1],[4,1]]
output: 2
```
解释：可使建筑高度为 `[0,1,2,1,2]`，最高建筑高度为 `2`。
## 示例2
```
input: n = 6, restrictions = []
output: 5
```
解释：无额外限制时，高度可依次为 `[0,1,2,3,4,5]`，最高为 `5`。
## 示例3
```
input: n = 10, restrictions = [[5,3],[2,5],[7,4],[10,3]]
output: 5
```
解释：可使建筑高度为 `[0,1,2,3,3,4,4,5,4,3]`，最高为 `5`。
# 解题思路
题目同时施加了三类约束：起点为 `0`、相邻高度差不超过 `1`、部分位置有高度上限。`n` 可达 `10^9`，无法逐栋枚举，需要找到少量「关键位置」集中处理。

## 从直观做法出发
最容易想到的是：对每一栋建筑 `i`，从左侧（建筑 `1` 高度 `0`）和右侧（建筑 `n` 无上限）分别推算它能达到的最大高度，再取两者较小值。但关键建筑有 `10^5` 个、总栋数有 `10^9` 个，逐栋计算显然不可行。

观察示例可以发现：高度上限只出现在 `restrictions` 给出的位置，其余建筑的高度由相邻关系和这些「卡点」共同决定。因此真正需要关心的，只有**有限个关键位置**——起点、终点，以及所有受限建筑。

## 关键观察：峰值只出现在相邻关键位置之间
任意合法高度序列中，全局最高点要么落在某个关键位置上，要么落在两个关键位置之间的某栋建筑上。

考虑相邻关键位置 $x_i$ 与 $x_{i+1}$，其间距离为 $d = x_{i+1} - x_i$。若位置 $x_i$ 处高度上界为 $h_i$，$x_{i+1}$ 处为 $h_{i+1}$，则这段区间内能抬起的最高「山峰」高度为：

$$
\text{peak} = \left\lfloor \frac{h_i + h_{i+1} + d}{2} \right\rfloor
$$

直觉上：从左侧以不超过 $h_i$ 的高度出发，以每步最多 $+1$ 的斜率向中间爬升，同时从右侧以不超过 $h_{i+1}$ 的高度以每步最多 $+1$ 向中间爬升，两坡在中点相遇时即得峰值。最终答案就是所有区间 peak 的最大值。

因此问题归结为两步：
1. 求出每个关键位置在**全局约束**下的高度上界 $h[i]$；
2. 枚举相邻关键位置对，用上式取最大值。

## 预处理：补全边界并排序
`restrictions` 中不含建筑 `1` 和建筑 `n`，但它们是天然的关键点。补入边界后统一排序：

- `[1, 0]`：建筑 `1` 高度固定为 `0`；
- `[n, inf]`：建筑 `n` 无高度上限（用 `inf` 表示）。

```python
restrictions += [[1, 0], [n, inf]]
restrictions.sort()
```

以示例 1（`n = 5, restrictions = [[2,1],[4,1]]`）为例，排序后得到：

| 下标 `i` | 位置 `x` | 限制 `limit` |
| --- | --- | --- |
| 0 | 1 | 0 |
| 1 | 2 | 1 |
| 2 | 4 | 1 |
| 3 | 5 | ∞ |

## 正向扫描：来自左侧的上界
从左向右，假设在位置 $x_{i-1}$ 处高度最多为 $h[i-1]$，则走到 $x_i$（距离 $x_i - x_{i-1}$）时，理论上限为 $h[i-1] + (x_i - x_{i-1})$，再与该位置的显式限制取小：

$$
h[i] = \min\bigl(h[i-1] + x_i - x_{i-1},\ \text{limit}_i\bigr)
$$

示例 1 正向扫描：

| `i` | `x` | 计算 | `h[i]` |
| --- | --- | --- | --- |
| 0 | 1 | 起点 | 0 |
| 1 | 2 | `min(0+1, 1)` | 1 |
| 2 | 4 | `min(1+2, 1)` | 1 |
| 3 | 5 | `min(1+1, ∞)` | 2 |

此时 `h[3]=2` 只考虑了从左而来的约束，尚未被右侧限制收紧。

## 反向扫描：来自右侧的上界
再从右向左，用对称逻辑收紧上界。在位置 $x_{i+1}$ 处高度最多为 $h[i+1]$，则位置 $x_i$ 处来自右侧的上界为 $h[i+1] + (x_{i+1} - x_i)$，与当前值取小：

$$
h[i] = \min\bigl(h[i],\ h[i+1] + x_{i+1} - x_i\bigr)
$$

示例 1 反向扫描：

| `i` | 计算 | `h[i]` |
| --- | --- | --- |
| 2 | `min(1, 2+1)` | 1 |
| 1 | `min(1, 1+2)` | 1 |
| 0 | `min(0, 1+1)` | 0 |

两轮扫描后，$h[i]$ 即为位置 $x_i$ 在**全部约束**下的高度上界。

## 段内取峰值
最后枚举每对相邻关键位置，套用峰值公式：

```python
max((x[i+1] - x[i] + h[i] + h[i+1]) // 2 for i in range(m - 1))
```

示例 1 各段：

| 区间 | `d` | `h[i]` | `h[i+1]` | peak |
| --- | --- | --- | --- | --- |
| 1 → 2 | 1 | 0 | 1 | `(1+0+1)//2 = 1` |
| 2 → 4 | 2 | 1 | 1 | `(2+1+1)//2 = 2` |
| 4 → 5 | 1 | 1 | 2 | `(1+1+2)//2 = 2` |

最大值为 `2`，与示例输出一致。

## 复杂度
- 时间复杂度：$O(m \log m)$，$m$ 为关键位置数（`restrictions.length + 2`），排序 dominates
- 空间复杂度：$O(m)$，存储高度上界数组

## 代码
```python
from math import inf

class Solution:
    def maxBuilding(self, n: int, restrictions: List[List[int]]) -> int:
        restrictions += [[1, 0], [n, inf]]
        restrictions.sort()

        m = len(restrictions)
        h = [0] * m

        for i in range(1, m):
            h[i] = min(h[i - 1] + restrictions[i][0] - restrictions[i - 1][0], restrictions[i][1])
        for i in range(m - 2, -1, -1):
            h[i] = min(h[i], h[i + 1] + restrictions[i + 1][0] - restrictions[i][0])

        return max(restrictions[i + 1][0] - restrictions[i][0] + h[i] + h[i + 1] for i in range(m - 1)) // 2
```

# 解题思路（斜率视角）
上一节的正向/反向扫描可以换一个更几何化的角度理解，有助于记住峰值公式从何而来。

相邻建筑高度差不超过 $1$，意味着高度曲线沿编号方向的**斜率绝对值**不超过 $1$——只能以每步 $\pm 1$ 的坡度升降。每个关键位置 $x_i$ 给出一个「天花板」$h[i]$，合法高度曲线必须始终位于这些天花板之下。

在区间 $[x_i, x_{i+1}]$ 上，曲线从左侧天花板 $h[i]$ 出发、以最大斜率 $+1$ 向中间爬升，同时从右侧天花板 $h[i+1]$ 以最大斜率 $+1$（等价于从左看是 $-1$）向中间爬升。两坡在中点相遇时达到最高，该点高度正是：

$$
\left\lfloor \frac{h[i] + h[i+1] + (x_{i+1} - x_i)}{2} \right\rfloor
$$

正向扫描等价于「只保留左坡天花板」，反向扫描等价于「再叠加右坡天花板取交集」。这与力扣相似题 [找到带限制序列的最大值](https://leetcode.cn/problems/find-maximum-value-in-a-constrained-sequence/) 的「双向松弛」本质相同，只是本题将松弛后的上界用于段内峰值而非直接输出某一点的值。

从渐近复杂度看，$O(m \log m)$ 已是最优——关键位置数 $m \le 10^5 + 2$，无法再降；也不存在比直接计算峰值更优的算法范式（例如二分答案会引入额外的 $\log H$ 因子，反而更慢）。

# 代码优化
本题时间复杂度已达最优，优化空间主要在边界处理与代码可读性。

## 合并位置 `n` 的重复条目
若 `restrictions` 中已包含 `[n, maxHeight]`，补入 `[n, inf]` 后排序会出现两个位置为 `n` 的条目。正确性不受影响（第二趟正向扫描距离为 `0`，`h` 不变），但可以在排序后合并：同一位置取 `limit` 的较小值。

```python
restrictions += [[1, 0], [n, inf]]
restrictions.sort()
merged = [restrictions[0]]
for x, lim in restrictions[1:]:
    if merged[-1][0] == x:
        merged[-1][1] = min(merged[-1][1], lim)
    else:
        merged.append([x, lim])
restrictions = merged
```

## 拆出位置与限制，减少二维下标
将 `restrictions` 拆为 `xs` 与 `limits` 两个数组，循环体更紧凑，也避免反复写 `restrictions[i][0]`：

```python
from math import inf

class Solution:
    def maxBuilding(self, n: int, restrictions: List[List[int]]) -> int:
        restrictions += [[1, 0], [n, inf]]
        restrictions.sort()
        xs = [x for x, _ in restrictions]
        limits = [lim for _, lim in restrictions]
        m = len(xs)
        h = [0] * m

        for i in range(1, m):
            h[i] = min(h[i - 1] + xs[i] - xs[i - 1], limits[i])
        for i in range(m - 2, -1, -1):
            h[i] = min(h[i], h[i + 1] + xs[i + 1] - xs[i])

        return max(xs[i + 1] - xs[i] + h[i] + h[i + 1] for i in range(m - 1)) // 2
```

语义与原写法完全一致，仅将「关键位置」与「高度上限」解耦，便于阅读与调试。
