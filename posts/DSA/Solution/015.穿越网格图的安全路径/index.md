---
title: 穿越网格图的安全路径
author: Hygen
description: LeetCode-3286
category: DSA
tags: [Solution, BFS, 矩阵]
published: 2026-07-02
---
# 题目描述
[题目链接](https://leetcode.cn/problems/find-a-safe-walk-through-a-grid/description/?envType=daily-question&envId=2026-07-02)

给你一个 $m \times n$ 的二维二进制数组 `grid` 和一个整数 `health`。

你从左上角 $(0, 0)$ 出发，目标是到达右下角 $(m-1, n-1)$。每次可以向**上、下、左、右**四个方向之一移动。

- 经过 `grid[i][j] = 0` 的格子是安全的，**不消耗**健康值；
- 经过 `grid[i][j] = 1` 的格子是危险的，**消耗 1 点**健康值。

返回是否能在**健康值始终为正**（即到达终点时健康值 $\ge 1$）的前提下到达终点。

## 输入
- `grid`：$m \times n$ 的二进制矩阵（元素值为 `0` 或 `1`）
- `health`：初始健康值（整数）

## 输出
- `bool`：能否安全到达终点

## 约束条件
1. `m == grid.length`
2. `n == grid[i].length`
3. `1 <= m, n <= 50`
4. `2 <= m * n`
5. `1 <= health <= m + n`
6. `grid[i][j]` 为 `0` 或 `1`

## 示例1
```
输入: grid = [[0,1,0,0,0],[0,1,0,1,0],[0,0,0,1,0]], health = 1
输出: true
```
解释：可以沿着如下路径（沿值为 `0` 的格子）安全到达终点，不消耗任何健康值，终点时健康值仍为 `1`。

## 示例2
```
输入: grid = [[0,1,1,0,0,0],[1,0,1,0,0,0],[0,1,1,1,0,1],[0,0,1,0,1,0]], health = 3
输出: false
```
解释：从起点到终点至少需要消耗 `4` 点健康值，初始 `health = 3` 不足以到达。

## 示例3
```
输入: grid = [[1,1,1],[1,0,1],[1,1,1]], health = 5
输出: true
```
解释：经过中间的安全格子 `(1, 1)`，累计消耗为 `4`，终点时健康值为 `5 - 4 = 1`，满足条件。

# 解题思路
问题本质是：在网格中寻找一条从起点到终点的路径，使得路径上所有格子的值之和（总消耗）**严格小于**初始健康值。这是一个典型的**最短路径**问题——每条边的权重就是目标格子的值（`0` 或 `1`），我们要找累计权重最小的路径。

> 🧠 **直觉理解**：把 `grid` 想象成一片沼泽地——值为 `0` 的格子是硬路面（不费体力），值为 `1` 的格子是泥潭（每踩一脚耗 1 点体力）。问题等价于：从左上走到右下，能否在体力耗尽前到达？

## 从暴力搜索出发
最直接的想法：用 DFS 枚举所有可能的路径，沿途累加消耗，到达终点时检查是否小于 `health`：

```python
class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]

        def dfs(x: int, y: int, cost: int) -> bool:
            if x == m - 1 and y == n - 1:
                return cost < health
            visited[x][y] = True
            for a, b in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                if 0 <= a < m and 0 <= b < n and not visited[a][b]:
                    new_cost = cost + grid[a][b]
                    if new_cost < health:  # 剪枝：已经超过 health 就不必继续
                        if dfs(a, b, new_cost):
                            return True
            visited[x][y] = False
            return False

        return dfs(0, 0, grid[0][0])
```

这种做法的时间复杂度是指数级的——每个格子都可能被多次访问，最坏情况下需要遍历所有 $4^{mn}$ 条可能的路径。在 $m, n \le 50$ 时完全不可行。但暴力解揭示了一件重要的事：**我们只关心到达每个格子的「最小累计消耗」，而不是路径本身**。

## 建模为最短路径问题
将每个格子视为图中的节点，相邻格子的移动视为有向边，边的权重等于**目标格子**的值 `grid[a][b]`。那么「到达终点的最小健康消耗」就是从起点 $(0, 0)$ 到终点 $(m-1, n-1)$ 的**最短路径权重**。

因为边权只有 `0`（安全格子）和 `1`（危险格子）两种，经典的 **0-1 BFS** 正好适用。

> 💡 **小贴士**：0-1 BFS 是 Dijkstra 算法在边权仅为 0 或 1 时的特化版本——用双端队列（deque）代替优先队列，将时间复杂度从 $O(V \log V)$ 优化到 $O(V)$。核心技巧：遇到权重 `0` 的边，将新节点插入队**首**（优先处理）；遇到权重 `1` 的边，插入队**尾**。

## 0-1 BFS 推演
维护一个距离矩阵 `d[i][j]`，表示从起点 $(0, 0)$ 到 $(i, j)$ 的最小累计消耗，初始化为无穷大，起点处 `d[0][0] = grid[0][0]`。

每次从 deque 前端取出一个格子 $(x, y)$，检查它的四个邻居 $(a, b)$：

- 如果通过 $(x, y)$ 到达 $(a, b)$ 的消耗 $d[x][y] + grid[a][b]$ 比已知的 $d[a][b]$ 更小，则更新 $d[a][b]$
- 若 `grid[a][b] == 0`（零消耗边），插入队首 → 下轮优先处理，确保同等消耗的格子被连续探索
- 若 `grid[a][b] == 1`（单位消耗边），插入队尾 → 等当前消耗层级处理完后再处理

> ⚠️ **注意**：0-1 BFS 要求从 deque **前端**取出节点（`popleft`），因为 `appendleft` 会打乱 BFS 层级顺序。这个细节决定了算法正确性——如果错误地从后端取出，就退化成了普通队列 BFS，无法保证最优。

以示例 1 的网格为例追踪算法执行（`health = 1`）：

```text
grid:
[0, 1, 0, 0, 0]
[0, 1, 0, 1, 0]
[0, 0, 0, 1, 0]
```

| 步骤 | 出队 `(x,y)` | `d[x][y]` | 更新的邻居 | 新 `d` 值 | 入队位置 |
|:--:|:--|:--:|------|:--:|:--:|
| 0 | — | — | 初始化 | `d[0][0]=0` | 队尾 `(0,0)` |
| 1 | `(0,0)` | 0 | `(1,0)` cost=0 | `d[1][0]=0` | 队首 |
| | | | `(0,1)` cost=1 | `d[0][1]=1` | 队尾 |
| 2 | `(1,0)` | 0 | `(2,0)` cost=0 | `d[2][0]=0` | 队首 |
| 3 | `(2,0)` | 0 | `(2,1)` cost=0 | `d[2][1]=0` | 队首 |
| 4 | `(2,1)` | 0 | `(2,2)` cost=0 | `d[2][2]=0` | 队首 |
| | | | `(1,1)` cost=1 | `d[1][1]=1` | 队尾 |
| 5 | `(2,2)` | 0 | `(2,3)` cost=1 | `d[2][3]=1` | 队尾 |
| | | | `(1,2)` cost=0 | `d[1][2]=0` | 队首 |
| 6 | `(1,2)` | 0 | `(0,2)` cost=0 | `d[0][2]=0` | 队首 |
| 7 | `(0,2)` | 0 | `(0,3)` cost=0 | `d[0][3]=0` | 队首 |
| 8 | `(0,3)` | 0 | `(0,4)` cost=0 | `d[0][4]=0` | 队首 |
| 9 | `(0,4)` | 0 | `(1,4)` cost=0 | `d[1][4]=0` | 队首 |
| 10 | `(1,4)` | 0 | `(2,4)` cost=0 | `d[2][4]=0` | 队首 |

最终 `d[2][4] = 0 < health = 1`，返回 `True`。整个过程队列始终优先处理 `cost=0` 的邻居，确保找到的路径是最优的。

## 复杂度
- 时间复杂度：$O(m \times n)$，每个格子最多入队一次（因为只在发现更优消耗时更新并入队，而 `d` 值的范围极小——每个格子只会被更新 $O(1)$ 次）
- 空间复杂度：$O(m \times n)$，距离矩阵 `d` 和 deque 的最大长度

## 代码
```python
from collections import deque
from math import inf
from typing import List


class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        d = [[inf] * n for _ in range(m)]
        d[0][0] = grid[0][0]

        q = deque([(0, 0)])
        while q:
            x, y = q.popleft()
            for a, b in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                if 0 <= a < m and 0 <= b < n:
                    if d[x][y] + grid[a][b] < d[a][b]:
                        d[a][b] = d[x][y] + grid[a][b]
                        if not grid[a][b]:
                            q.appendleft((a, b))
                        else:
                            q.append((a, b))
        return d[-1][-1] < health
```

代码的几个关键点：
- `d[i][j]` 初始化为 `inf`，确保任何真实路径都能首次更新它
- `d[0][0] = grid[0][0]`：起点的格子值也算入消耗
- `grid[a][b]` 为 `0` 时 `appendleft`（优先处理），为 `1` 时 `append`（延后处理）
- 最终判断 `d[-1][-1] < health`：消耗必须**严格小于**初始健康值，因为到达终点时健康值需 $\ge 1$

# 解题思路（Dijkstra）
如果网格中的值不是仅限 `0/1` 而是任意非负整数，0-1 BFS 就无法直接使用了。此时需要用通用的 Dijkstra 算法——用小顶堆（优先队列）维护「当前已知的到各格子的最短距离」，每次取出距离最小的节点进行松弛。

```python
from heapq import heappop, heappush
from math import inf


class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        d = [[inf] * n for _ in range(m)]
        d[0][0] = grid[0][0]

        pq = [(d[0][0], 0, 0)]  # (cost, x, y)
        while pq:
            cost, x, y = heappop(pq)
            if cost > d[x][y]:
                continue  # 过期数据，跳过
            if x == m - 1 and y == n - 1:
                return cost < health
            for a, b in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                if 0 <= a < m and 0 <= b < n:
                    new_cost = cost + grid[a][b]
                    if new_cost < d[a][b]:
                        d[a][b] = new_cost
                        heappush(pq, (new_cost, a, b))
        return False
```

| 对比项 | 0-1 BFS | Dijkstra |
| --- | --- | --- |
| 时间 | $O(mn)$ | $O(mn \log(mn))$ |
| 空间 | $O(mn)$ | $O(mn)$ |
| 数据结构 | 双端队列 `deque` | 小顶堆 `heapq` |
| 适用场景 | 边权仅 $0$ 或 $1$ | 边权为任意非负整数 |
| 代码简洁度 | 稍简洁 | 也较简洁 |

本题 `grid` 的值只有 `0` 和 `1`，0-1 BFS 是最优选择。Dijkstra 也能过（$50 \times 50 = 2500$ 个节点，$\log$ 开销很小），但面试中展示 0-1 BFS 能体现对边权特殊性质的洞察。

# 代码优化
0-1 BFS 已经是本题的最优算法，以下聚焦于工程细节。

## 提前返回
一旦从 deque 中取出的节点是终点 `(m-1, n-1)`，`d[m-1][n-1]` 已经是最小消耗——因为 0-1 BFS 保证节点首次出队时 `d` 值即为最短距离（类似 Dijkstra）。可以直接返回而不必跑完整个 BFS：

```python
while q:
    x, y = q.popleft()
    if x == m - 1 and y == n - 1:
        return d[x][y] < health
    for a, b in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
        ...
```

## 方向数组提取
将四个方向的偏移量提取为常量，减少 for 循环中的元组构造开销（微优化，但对竞赛场景有意义）：

```python
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
...
for dx, dy in DIRS:
    a, b = x + dx, y + dy
    ...
```

## 最终优化版
```python
from collections import deque
from math import inf


class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        m, n = len(grid), len(grid[0])
        d = [[inf] * n for _ in range(m)]
        d[0][0] = grid[0][0]

        q = deque([(0, 0)])
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while q:
            x, y = q.popleft()
            if x == m - 1 and y == n - 1:
                return d[x][y] < health
            for dx, dy in DIRS:
                a, b = x + dx, y + dy
                if 0 <= a < m and 0 <= b < n:
                    nd = d[x][y] + grid[a][b]
                    if nd < d[a][b]:
                        d[a][b] = nd
                        if grid[a][b]:
                            q.append((a, b))
                        else:
                            q.appendleft((a, b))
        return False
```

## 小结
| 写法 | 优点 | 场景 |
| --- | --- | --- |
| 原版（全遍历后判断） | 最简洁，逻辑直白 | 日常刷题、不追求极致性能 |
| 提前返回版 | 大多数情况能提前结束 | 网格较大、终点附近安全格子多时 |
| 方向数组版 | 常数优化，扩展性好 | 竞赛、八方向变体等扩展场景 |

本题 $m, n \le 50$，三种写法的实际运行时间差异在毫秒级，选择自己最熟悉的即可。
