---
title: 雪糕的最大数量
author: Hygen
description: LeetCode-1833
category: Algorithm and Structure
tags: [Solution, 贪心, 排序]
published: 2026-06-21
---
# 题目描述
[题目链接](https://leetcode.cn/problems/maximum-ice-cream-bars/?envType=daily-question&envId=2026-06-21)

夏日炎炎，小男孩 Tony 想买一些雪糕消消暑。

商店中新到 `n` 支雪糕，用长度为 `n` 的数组 `costs` 表示雪糕的定价，其中 `costs[i]` 表示第 `i` 支雪糕的现金价格。Tony 一共有 `coins` 现金可以用于消费，他想要买尽可能多的雪糕。

**注意：** Tony 可以按任意顺序购买雪糕。

给你价格数组 `costs` 和现金量 `coins`，请你计算并返回 Tony 用 `coins` 现金能够买到的雪糕的**最大数量**。

## 输入
整数数组 `costs`，整数 `coins`
## 输出
能买到的雪糕最大数量
## 约束条件
1. `1 <= costs.length == n <= 10^5`
2. `1 <= costs[i] <= 10^5`
3. `1 <= coins <= 10^8`
## 示例1
```
input: costs = [1,3,2,4,1], coins = 7
output: 4
```
解释：Tony 可以买下标为 0、1、2、4 的雪糕，总价为 `1 + 3 + 2 + 1 = 7`。
## 示例2
```
input: costs = [10,6,8,7,7,8], coins = 5
output: 0
```
解释：Tony 没有足够的钱买任何一支雪糕。
## 示例3
```
input: costs = [1,6,3,1,2,5], coins = 20
output: 6
```
解释：Tony 可以买下所有的雪糕，总价为 `1 + 6 + 3 + 1 + 2 + 5 = 18`。
# 解题思路
题目只要求**数量最多**，且 Tony 可以按任意顺序购买。在总花费不超过 `coins` 的前提下，怎样选才能买到最多支雪糕？

## 从直观做法出发
最容易想到的是：枚举所有子集或排列，找出花费不超过 `coins` 且数量最大的那一组。但 `n` 可达 `10^5`，子集规模是指数级，显然不可行。

换一个角度：每支雪糕的价格不同，而目标只是「多买几支」。在钱有限时，直觉上应该**优先买便宜的**——同样的钱，买便宜的能多换几支。

## 贪心：先买最便宜的
形式化地说：若存在两支雪糕，价格分别为 `a < b`，而当前方案先买了价格为 `b` 的那支、却跳过了价格为 `a` 的，那么把这次购买换成 `a`，会省下 `b - a` 的现金，后续至少不会比原来更差（多出来的钱还能继续买别的）。因此，最优方案里一定不会出现「放着更便宜的没买，却买了更贵的」这种情况。

结论：**按价格从低到高依次购买**，能买到的就买，买不到就停——这就是贪心策略。

## 排序后线性扫描
要实现「从便宜到贵」，先把 `costs` 升序排序，再从左到右遍历：若当前剩余现金 `coins` 够付这支雪糕的价格 `x`，就买下来（`ans += 1`，`coins -= x`）；否则后面的价格只会更高，直接结束。

以 `costs = [1,3,2,4,1], coins = 7` 为例，排序后 `costs = [1,1,2,3,4]`：

| 价格 `x` | `coins`（买前） | 能否购买 | `ans` | `coins`（买后） |
| --- | --- | --- | --- | --- |
| 1 | 7 | 是 | 1 | 6 |
| 1 | 6 | 是 | 2 | 5 |
| 2 | 5 | 是 | 3 | 3 |
| 3 | 3 | 是 | 4 | 0 |
| 4 | 0 | 否 | 4 | 0 |

最终 `ans = 4`，与示例输出一致。

## 复杂度
- 时间复杂度：$O(n \log n)$，排序 dominates；扫描为 $O(n)$
- 空间复杂度：$O(1)$ 或 $O(\log n)$（取决于排序实现是否原地）

## 代码
```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        ans = 0
        for i, x in enumerate(costs):
            if coins >= x:
                ans += 1
                coins -= x
        return ans
```

# 解题思路（计数排序）
上一节用内置 `sort` 完成排序，复杂度 $O(n \log n)$。观察约束：`costs[i]` 取值范围为 $[1, 10^5]$，是一个**有界整数**——此时可以用**计数排序**在 $O(n + V)$ 时间内完成「按价格分组」，其中 $V$ 为价格上界。

## 为何计数排序更合适
比较排序的下界是 $O(n \log n)$，与元素取值范围无关。当 $V = \max(\text{costs}) \le 10^5$ 且 $n$ 也约为 $10^5$ 时，$O(n + V)$ 与 $O(n \log n)$ 同阶，但常数更小；若 $V \ll n$（价格种类少），计数排序优势更明显。

力扣本题也提示「必须使用计数排序」，核心思想正是：价格范围有限，不必对 $n$ 个元素做通用比较排序。

## 按价格桶批量购买
先统计每个价格 `p` 出现了多少次 `cnt[p]`，再从 `p = 1` 到 `max(costs)` 枚举：

- 若买光所有价格为 `p` 的雪糕只需 `p * cnt[p]`，且 `coins` 够付，则全部买下；
- 否则，用剩余现金能买 `coins // p` 支，买完即停。

价格高于 `coins` 的桶无需再看（买一支都不够），枚举上界可截断为 `min(max(costs), coins)`。

以 `costs = [1,3,2,4,1], coins = 7` 为例，计数后 `cnt[1]=2, cnt[2]=1, cnt[3]=1, cnt[4]=1`：

| 价格 `p` | `cnt[p]` | 全买花费 | `coins`（买前） | 操作 | `ans` | `coins`（买后） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2 | 7 | 全买 2 支 | 2 | 5 |
| 2 | 1 | 2 | 5 | 买 1 支 | 3 | 3 |
| 3 | 1 | 3 | 3 | 买 1 支 | 4 | 0 |
| 4 | 1 | 4 | 0 | 买不起，停止 | 4 | 0 |

答案仍为 `4`。贪心逻辑与排序扫描完全一致，只是把「逐支遍历」换成了「按价格桶批量处理」。

## 复杂度
- 时间复杂度：$O(n + \min(V, \text{coins}))$，统计 $O(n)$，枚举价格桶 $O(\min(V, \text{coins}))$
- 空间复杂度：$O(V)$，计数数组

# 代码优化
排序贪心已是正确且简洁的写法，优化空间主要在提前终止、写法精简，以及结合计数数组做批量购买。

## 买不起时提前 `break`
排序后价格单调不降，一旦 `coins < x`，后面的雪糕一定也买不起，无需继续循环：

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        costs.sort()
        ans = 0
        for x in costs:
            if coins < x:
                break
            ans += 1
            coins -= x
        return ans
```

## 去掉无用的 `enumerate`
循环中并未使用下标 `i`，直接 `for x in costs` 更干净，语义也更贴近「按价格从低到高依次尝试」。

## 计数数组 + 批量购买（常数更优）
当同一价格出现多次时，逐支扣款会产生多余循环。用计数数组按桶处理，每轮用 `min(cnt[p], coins // p)` 一次算清能买几支：

```python
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        mx = max(costs)
        cnt = [0] * (mx + 1)
        for x in costs:
            cnt[x] += 1

        ans = 0
        for p in range(1, min(mx, coins) + 1):
            if cnt[p] == 0:
                continue
            take = min(cnt[p], coins // p)
            ans += take
            coins -= take * p
            if coins == 0:
                break
        return ans
```

与排序版贪心等价，在价格重复较多时循环次数更少；渐近复杂度为 $O(n + \min(V, \text{coins}))$，是本题在思路层面更贴合数据范围的写法。
