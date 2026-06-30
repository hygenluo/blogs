---
title: 气球的最大数量
author: Hygen
description: LeetCode-1189
category: Algorithm and Structure
tags: [Solution, 哈希, 字符串]
published: 2026-06-22
---
# 题目描述
[题目链接](https://leetcode.cn/problems/maximum-number-of-balloons/description/?envType=daily-question&envId=2026-06-22)

给你一个字符串 `text`，你可以使用其中的字符来组成尽可能多的单词 `"balloon"`。

给你一个字符串 `text`，计算并返回能够拼成 `"balloon"` 的**最大数量**。

## 输入
字符串 `text`
## 输出
能拼出的 `"balloon"` 最大数量
## 约束条件
1. `1 <= text.length <= 10^4`
2. `text` 仅由小写英文字母组成
## 示例1
```
input: text = "nlaebolko"
output: 1
```
解释：用 `text` 中的字符可以拼出一个 `"balloon"`。
## 示例2
```
input: text = "loonbalxballpoon"
output: 2
```
解释：可以拼出两个 `"balloon"`，剩余字符无法继续拼出第三个。
## 示例3
```
input: text = "leetcode"
output: 0
```
解释：`text` 中没有字母 `b`，无法拼出任何 `"balloon"`。
# 解题思路
题目要求用 `text` 中的字符拼出尽可能多的 `"balloon"`。每个字符只能使用一次，且拼出的单词必须恰好是 `"balloon"`。

## 从直观做法出发
最容易想到的是：在 `text` 里反复「找齐」一组 `b-a-l-l-o-o-n`，凑齐就 `ans += 1`，直到再也凑不齐为止。但每凑一组都要在剩余字符里重新扫描、匹配，实现繁琐，也容易在重复字母上出错。

换一个角度：与其模拟「一组一组地拼」，不如先问——**拼一个 `"balloon"` 到底需要哪些字母、各要几个？** 答案一旦固定，整道题就变成了「资源够不够」的问题。

## 拆解单词 balloon
把 `"balloon"` 的字母拆开统计：

| 字母     | `b` | `a` | `l` | `o` | `n` |
| -------- | --- | --- | --- | --- | --- |
| 出现次数 | 1   | 1   | 2   | 2   | 1   |

也就是说，每拼出 **1** 个 `"balloon"`，就要消耗 `1` 个 `b`、`1` 个 `a`、`2` 个 `l`、`2` 个 `o`、`1` 个 `n`。

## 统计字母频次
因此第一步很直接：遍历 `text`，统计每个字母出现了多少次。Python 里用 `Counter(text)` 一行即可完成。

以 `text = "loonbalxballpoon"` 为例，统计结果为：

| 字母 | `b` | `a` | `l` | `o` | `n` | 其他           |
| ---- | --- | --- | --- | --- | --- | -------------- |
| 频次 | 2   | 2   | 4   | 4   | 2   | `x`: 1, `p`: 1 |

## 木桶原理：取最小值
有了各字母的存量，能拼几个 `"balloon"` 取决于**最紧缺**的那类字母——就像木桶能装多少水，取决于最短的那块板。

每种字母能「支撑」的 balloon 数量如下：

- `b`：每个 balloon 用 1 个 → 最多 `cnt['b']` 个
- `a`：每个 balloon 用 1 个 → 最多 `cnt['a']` 个
- `l`：每个 balloon 用 2 个 → 最多 `cnt['l'] // 2` 个
- `o`：每个 balloon 用 2 个 → 最多 `cnt['o'] // 2` 个
- `n`：每个 balloon 用 1 个 → 最多 `cnt['n']` 个

最终答案就是上述五个值中的**最小值**：

$$\text{ans} = \min\big(\text{cnt['b']},\ \text{cnt['a']},\ \text{cnt['l']} // 2,\ \text{cnt['o']} // 2,\ \text{cnt['n']}\big)$$

继续以上面的例子：`min(2, 2, 4//2, 4//2, 2) = min(2, 2, 2, 2, 2) = 2`，与示例输出一致。

`Counter` 对不存在的键会返回 `0`（例如 `text = "leetcode"` 中没有 `b`，`cnt['b']` 为 `0`），因此 `min` 自然得到 `0`，无需额外判空。

## 复杂度
- 时间复杂度：$O(n)$，$n$ 为 `text` 长度，统计字母频次需遍历一次
- 空间复杂度：$O(1)$，至多统计 26 个小写字母

## 代码
```python
from collections import Counter

class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        cnt = Counter(text)
        return min(cnt['b'], cnt['a'], cnt['l'] // 2, cnt['o'] // 2, cnt['n'])
```

# 解题思路（只统计相关字母）
上一节用 `Counter(text)` 统计了 `text` 中**所有**字母的频次。但拼 `"balloon"` 只会用到 `b`、`a`、`l`、`o`、`n` 这五个字母——`text` 里其余的字符（如示例中的 `x`、`p`）对答案没有任何贡献。

## 为何可以忽略无关字母
拼单词时，多出来的字母既不能被「存起来」给下次用，也不会参与任何一次 `"balloon"` 的组成。因此统计阶段只需关心这五个目标字母，其余字符完全可以跳过。

这样做的好处是：

- **思路上更清晰**：问题被精确归结为「五种资源的配额」；
- **常数更小**：不必为 26 个字母都维护计数（尤其当 `text` 很长、字母种类很多时）。

## 单次遍历，按需累加
维护一个长度为 5 的计数器（或用字典映射），遍历 `text` 时，若当前字符属于 `{b, a, l, o, n}` 才累加，否则直接跳过。遍历结束后，同样用木桶原理取 `min` 即可。

以 `text = "leetcode"` 为例，五个相关字母的计数全为 `0`，`min` 结果为 `0`；以 `text = "nlaebolko"` 为例，统计得 `b=1, a=1, l=2, o=2, n=1`，`min(1, 1, 1, 1, 1) = 1`。

逻辑与上一节完全一致，区别仅在于「统计谁」——从「全体字母」收窄为「目标字母」。

# 代码优化
`Counter` 写法已经足够简洁正确。若希望在代码层面进一步精简或避免依赖 `collections`，可以从以下方向优化。

## 用固定需求表 + 循环代替手写 `min`
把每种字母的需求写进字典，用循环统一计算「该字母最多能支撑几个 balloon」，避免在 `return` 里重复写五个参数：

```python
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        need = {'b': 1, 'a': 1, 'l': 2, 'o': 2, 'n': 1}
        cnt = {c: 0 for c in need}
        for c in text:
            if c in cnt:
                cnt[c] += 1
        return min(cnt[c] // need[c] for c in need)
```

字母种类固定为 5，循环开销可忽略；若日后单词变化，只需改 `need` 字典，扩展性更好。

## 用长度为 26 的数组代替字典
字母均为小写，`ord(c) - ord('a')` 可作下标，用整型数组计数，访问比字典略快：

```python
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        cnt = [0] * 26
        for c in text:
            cnt[ord(c) - ord('a')] += 1
        return min(cnt[ord(c) - ord('a')] // need
                     for c, need in [('b', 1), ('a', 1), ('l', 2), ('o', 2), ('n', 1)])
```

渐近复杂度仍为 $O(n)$ 时间、$O(1)$ 空间，适合对常数敏感的场景。

## 小结
| 写法                   | 优点             | 适用场景               |
| ---------------------- | ---------------- | ---------------------- |
| `Counter` + 手写 `min` | 最短、最易读     | 日常刷题、面试快速作答 |
| 需求表 + 循环          | 易扩展、逻辑集中 | 单词组成可能变化时     |
| 数组计数               | 常数更小         | 对性能有要求的实现     |

本题数据规模 $n \le 10^4$，三种写法差距极小；首选 `Counter` 版本即可，其余作为代码风格与扩展性的备选。
