---
title: 接雨水
author: Hygen
description: Leetcode-42
category: hot100
tags: [Solution, 双指针, 单调栈]
published: 2026-06-28
---
# 题目描述
[题目链接](https://leetcode.cn/problems/trapping-rain-water/description/?envType=study-plan-v2&envId=top-100-liked)

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
## 输入
整数数组 `height`
## 输出
能接的雨水总量（整数）
## 约束条件
1. `n == height.length`
2. `1 <= n <= 2 * 10^4`
3. `0 <= height[i] <= 10^5`
# 进阶约束
1. 能否将时间复杂度优化到 $O(n)$？
2. 能否将额外空间复杂度优化到 $O(1)$？
## 示例1
```
input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
output: 6
```
解释：蓝色部分为接住的雨水，共 `6` 个单位。
## 示例2
```
input: height = [4,2,0,3,2,5]
output: 9
```
# 解题思路
题目本质是：对每个位置，求它上方能「兜住」多少水。某一格的水位由**左侧最高柱**和**右侧最高柱**中的**较小值**决定，再减去自身高度：

$$\text{water}[i] = \max\bigl(0,\ \min(\text{leftMax}[i],\ \text{rightMax}[i]) - \text{height}[i]\bigr)$$

## 从暴力枚举出发
最直观的做法是：对每个位置 `i`，分别向左、向右扫描，求出左侧最大值和右侧最大值，再按上式累加：

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        n = len(height)
        for i in range(n):
            left_max = max(height[:i + 1])
            right_max = max(height[i:])
            ans += min(left_max, right_max) - height[i]
        return ans
```

思路正确，但每个位置都要左右各扫一遍，时间复杂度 $O(n^2)$，在 $n$ 可达 $2 \times 10^4$ 时会超时。瓶颈在于**重复计算**左右最大值——能否预处理出来？

## 前缀最大值：逐列算积水
定义两个数组：

- `pre[i]`：下标 `0` 到 `i`（含）中的最大高度，即位置 `i` 左侧（含自身）的最高柱
- `suf[i]`：下标 `i` 到 `n-1`（含）中的最大高度，即位置 `i` 右侧（含自身）的最高柱

一次从左到右、一次从右到左即可 $O(n)$ 预处理，再逐位累加：

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        pre = [0] * n
        suf = [0] * n
        pre[0] = height[0]
        for i in range(1, n):
            pre[i] = max(pre[i - 1], height[i])
        suf[-1] = height[-1]
        for i in range(n - 2, -1, -1):
            suf[i] = max(suf[i + 1], height[i])
        return sum(min(pre[i], suf[i]) - height[i] for i in range(n))
```

时间 $O(n)$，但用了 $O(n)$ 额外数组。进阶约束要求 $O(1)$ 空间——能否在遍历过程中**动态维护**左右最大值，而不显式存下整个数组？

## 双指针：只维护「有效」的一侧最大值
用两个指针 `l`、`r` 分别从数组首尾向中间移动，同时维护：

- `pre`：已处理过的**左半部分**的最大高度（不含当前 `l` 即将纳入的那一格之前的含义，代码里是先更新再累加）
- `suf`：已处理过的**右半部分**的最大高度

核心观察：当 `height[l] < height[r]` 时，**右侧**至少有一根高度为 `height[r]` 的柱子，它不低于 `height[l]`。因此对当前位置 `l` 而言，决定水位的「短板」一定在**左侧**——水位由 `pre`（左半已扫描部分的最大值）决定，与右侧更高细节无关。

反之，当 `height[l] >= height[r]` 时，左侧已有足够高的墙，当前位置 `r` 的水位由 `suf` 决定。

以 `height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]` 为例，前几步如下：

| `l` | `r` | `height[l]` | `height[r]` | 移动侧 | `pre` / `suf` | 本次积水 | `ans` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | 0 | 1 | 左（更矮） | `pre=0` | 0 | 0 |
| 1 | 11 | 1 | 1 | 右（相等移右） | `suf=1` | 0 | 0 |
| 1 | 10 | 1 | 2 | 左（更矮） | `pre=1` | 0 | 0 |
| 2 | 10 | 0 | 2 | 左（更矮） | `pre=1` | 1 | 1 |
| 3 | 10 | 2 | 2 | 右 | `suf=2` | 0 | 1 |
| 3 | 9 | 2 | 1 | 右（更矮） | `suf=2` | 1 | 2 |

每次移动较短（或相等时移动右）一侧，用该侧已维护的最大值减去当前高度，累加到答案。两侧指针最终相遇，所有位置的积水都算完。

为什么不会漏算？每当处理 `l` 时，右侧存在 `height[r]` 作为「可靠高墙」，左侧最大值 `pre` 就是该位置真实的 `leftMax`；对称地，处理 `r` 时左侧 `height[l]` 保证了 `suf` 的有效性。

## 复杂度
- 时间复杂度：$O(n)$，两个指针各最多移动 $n - 1$ 次
- 空间复杂度：$O(1)$，仅维护常数个变量

## 代码
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = 0
        pre = suf = 0

        while l < r:
            if height[l] < height[r]:
                pre = max(pre, height[l])
                ans += pre - height[l]
                l += 1
            else:
                suf = max(suf, height[r])
                ans += suf - height[r]
                r -= 1

        return ans
```

# 解题思路（单调栈）
双指针是**按列**逐格累加积水；另一种视角是**按层**横向「填水」——当遇到一根比栈顶更高的柱子时，之前凹下去的区域可以被填满一层。

维护一个**单调递减栈** `st`，存下标，栈底到栈顶对应的高度单调不增。遍历到位置 `i`、高度 `h` 时：

1. 若 `h` 不大于栈顶高度，说明还没形成新的「盆」，将 `i` 入栈
2. 若 `h` 大于栈顶高度，说明右侧出现了更高的墙，可以兜住水：
   - 弹出栈顶作为**盆底** `bottom`
   - 若栈空，说明左侧没有墙，无法接水，`break`
   - 否则栈顶（弹出后新的栈顶）是**左侧墙**，与当前 `h`（右侧墙）共同夹住一层水
   - 宽度为 `i - st[-1] - 1`，高度为 `min(左墙, 右墙) - bottom`

以 `height = [0, 1, 0, 2, 1, 0, 1, 3]` 为例，当 `i = 7`、`h = 3` 时，栈中依次为下标 `1(高1)、2(高0)、5(高0)、6(高1)`。连续弹出后，会在 `(1, 3)` 之间、`(6, 3)` 之间等位置分别算出一层积水。

这种写法把「凹槽」拆成若干横向矩形累加，与双指针的纵向切片殊途同归。

## 复杂度
- 时间复杂度：$O(n)$，每个下标最多入栈、出栈各一次
- 空间复杂度：$O(n)$，栈中最多存 $n$ 个下标

## 代码
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        st = []
        ans = 0
        
        for i, h in enumerate(height):
            while st and height[st[-1]] < h:
                bottom = height[st.pop()]
                if not st: break
                ans += (min(height[st[-1]], h) - bottom) * (i - st[-1] - 1)
            st.append(i)

        return ans
```

| 对比项 | 双指针 | 单调栈 |
| --- | --- | --- |
| 时间 | $O(n)$ | $O(n)$ |
| 空间 | $O(1)$ | $O(n)$ |
| 视角 | 逐列：每格水位由左右最大值决定 | 逐层：弹出盆底后横向填水 |
| 适用 | 本题最优解，代码短 | 便于扩展到接雨水 II 等二维变形 |

面试和竞赛中更推荐双指针：空间更优、实现更短；单调栈则适合需要**横向分层**或推广到矩阵接水的场景。

# 代码优化
本题在算法层面，$O(n)$ 时间已是下界（至少要看一遍数组），$O(1)$ 空间的双指针也是最优之一。因此优化空间主要在**代码简洁度**与**常数细节**。

## 合并 `pre` 与 `suf` 为单一变量
同一时刻只有较短一侧在被处理，另一侧的最大值变量本轮不会用到。可以只保留一个 `mx`，在左右分支里分别更新，语义不变：

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        ans = mx = 0
        while l < r:
            if height[l] < height[r]:
                mx = max(mx, height[l])
                ans += mx - height[l]
                l += 1
            else:
                mx = max(mx, height[r])
                ans += mx - height[r]
                r -= 1
        return ans
```

注意：两侧交替处理时 `mx` 的含义会随分支切换，但**每次累加前**它代表的正是当前侧所需的「已扫描最大值」，与分开维护 `pre`、`suf` 等价。若更看重可读性，保留 `pre`、`suf` 两个变量更清晰。

## 单调栈：提前缓存栈顶高度
内层 `while` 中多次访问 `height[st[-1]]`，可先取出栈顶下标与高度，减少重复下标计算：

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        st = []
        ans = 0
        for i, h in enumerate(height):
            while st:
                top = st[-1]
                if height[top] >= h:
                    break
                bottom = height[st.pop()]
                if not st:
                    break
                ans += (min(height[st[-1]], h) - bottom) * (i - st[-1] - 1)
            st.append(i)
        return ans
```

逻辑与原版相同，将 `height[st[-1]] < h` 的判断改为显式 `break`，对部分读者更易跟进循环出口。
