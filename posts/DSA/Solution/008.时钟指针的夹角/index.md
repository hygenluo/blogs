---
title: 时钟指针的夹角
author: Hygen
description: LeetCode-1344
category: Algorithm and Structure
tags: [Solution, 数学, 模拟]
published: 2026-06-18
---
# 题目描述
[题目链接](https://leetcode.cn/problems/angle-between-hands-of-a-clock/?envType=daily-question&envId=2026-06-18)

给你两个数 `hour` 和 `minutes`。请你返回在时钟上，由给定时间的时针和分针组成的**较小角**的角度（60 单位制，即一圈 360°）。

## 输入
整数 `hour`，整数 `minutes`
## 输出
时针与分针之间的较小夹角（浮点数）
## 约束条件
1. `1 <= hour <= 12`
2. `0 <= minutes <= 59`
3. 与标准答案误差在 `10^-5` 以内的结果都被视为正确结果
## 示例1
```
input: hour = 12, minutes = 30
output: 165
```
## 示例2
```
input: hour = 3, minutes = 30
output: 75
```
## 示例3
```
input: hour = 3, minutes = 15
output: 7.5
```
## 示例4
```
input: hour = 4, minutes = 50
output: 155
```
## 示例5
```
input: hour = 12, minutes = 0
output: 0
```
# 解题思路
题目要的是时针与分针之间的**较小夹角**，而不是有向角或较大那个角。核心在于：先把两根指针各自转过的角度算出来，再求它们之间的差，最后取「锐角/钝角中较小的那个」。

## 从表盘刻度出发
标准时钟一圈 360°，均匀分成 12 个大格、60 个小格：

- **分针**：每走 1 分钟转 $360° / 60 = 6°$
- **时针**：每走 1 小时转 $360° / 12 = 30°$；同时，时针会随分钟缓慢移动——每过 1 分钟再转 $30° / 60 = 0.5°$

因此给定 `minutes` 和 `hour` 后，两根指针相对于 12 点方向（顺时针）的角度分别为：

$$
m = 6 \times \text{minutes}, \quad h = \text{hour} \times 30 + \frac{\text{minutes}}{2}
$$

以示例 3（`hour = 3, minutes = 15`）为例：

| 指针 | 计算 | 角度 |
| --- | --- | --- |
| 分针 | $6 \times 15$ | $90°$ |
| 时针 | $3 \times 30 + 15/2$ | $97.5°$ |

两根指针几乎重合，夹角应为 $7.5°$。

## 求两针夹角
有了 $h$ 和 $m$，两针的夹角就是它们角度之差的绝对值。但直接写 `abs(h - m)` 在 $h < m$ 时虽然仍正确，用「补角」形式可以统一处理，避免显式分支：

$$
\text{ans} = (h + 360 - m) \bmod 360
$$

这相当于把 $h - m$ 归一化到 $[0, 360)$：若 $h \ge m$，结果就是 $h - m$；若 $h < m$，则加上 360 再取模，等价于从分针沿顺时针方向走到时针的角度。

仍以示例 3 验证：$\text{ans} = (97.5 + 360 - 90) \bmod 360 = 7.5$。

## 取较小角
表盘上两个方向的角度之和恒为 360°，题目要的是**较小**的那个，因此：

$$
\text{result} = \min(\text{ans},\ 360 - \text{ans})
$$

示例 1（`hour = 12, minutes = 30`）：$m = 180°$，$h = 12 \times 30 + 15 = 375°$，$\text{ans} = (375 + 360 - 180) \bmod 360 = 195°$，较小角为 $\min(195, 165) = 165°$。

示例 5（`hour = 12, minutes = 0`）：$m = 0°$，$h = 360°$，$\text{ans} = 0°$，两针重合，返回 $0$。

## 复杂度
- 时间复杂度：$O(1)$，固定几次算术运算
- 空间复杂度：$O(1)$

## 代码
```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        m = 6 * minutes
        h = hour * 30 + minutes / 2
        ans = (h + 360 - m) % 360
        return min(ans, 360 - ans)
```

# 解题思路（相对角速度）
上面的做法先分别求出两根指针的**绝对角度**，再作差。换一个视角：只关心两针之间的**相对位移**，可以跳过「各自归位到 12 点」这一步。

分针每分钟转 $6°$，时针每分钟转 $0.5°$，相对角速度为 $6 - 0.5 = 5.5°$/分钟。因此，从整点出发，分钟带来的「追赶量」是 $5.5 \times \text{minutes}$；小时刻度带来的初始差距是 $\text{hour} \times 30°$（`hour = 12` 时等价于 0）。

两针夹角的绝对差可直接写为：

$$
\left| \text{hour} \times 30 - \text{minutes} \times 5.5 \right|
$$

再同样取 $\min(\text{diff},\ 360 - \text{diff})$ 即可。思路与上一节等价，只是把「分针角度」和「时针因分钟产生的偏移」合并成了一项 $5.5 \times \text{minutes}$，心算时更紧凑。

# 代码优化
本题时间复杂度已是 $O(1)$，不存在算法层面的更优渐近复杂度。优化空间主要在代码简洁性与边界表达的清晰度。

## 用 `hour % 12` 处理 12 点
`hour = 12` 在表盘上与 0 点重合。当前写法中 $h = 12 \times 30 = 360°$，取模后结果仍正确，但语义上 12 点应视为第 0 格。写成 `hour % 12` 更直观，也避免 $h$ 出现 360、390 这类「绕了一圈」的数值：

```python
h = hour % 12 * 30 + minutes / 2
```

## 合并为单行表达式
若更在意代码紧凑，可将相对角速度公式与取较小角合并（需配合 `hour % 12`）：

```python
class Solution:
    def angleClock(self, hour: int, minutes: int) -> float:
        diff = abs(hour % 12 * 30 - minutes * 5.5)
        return min(diff, 360 - diff)
```

与原写法相比，变量更少、一行出结果；代价是 `(h + 360 - m) % 360` 那种「不用 `abs`、统一归一化到 $[0, 360)$」的写法在可读性上略胜一筹——可按个人偏好选择。
