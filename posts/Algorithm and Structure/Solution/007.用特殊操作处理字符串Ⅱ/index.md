---
title: 用特殊操作处理字符串Ⅱ
author: Hygen
description: LeetCode-3614
category: Algorithm and Structure
tags: [Solution, 字符串, 模拟]
published: 2026-06-17
---
# 题目描述
[题目链接](https://leetcode.cn/problems/process-string-with-special-operations-ii/description/?envType=daily-question&envId=2026-06-17)

给你一个字符串 `s`，由小写英文字母和特殊字符 `*`、`#` 和 `%` 组成。

同时给你一个整数 `k`。

请根据以下规则从左到右处理 `s` 中的字符，构造一个新的字符串 `result`：

- 如果字符是**小写**英文字母，则将其添加到 `result` 中。
- 字符 `'*'` 会**删除** `result` 中的最后一个字符（如果存在）。
- 字符 `'#'` 会**复制**当前的 `result` 并**追加**到其自身后面。
- 字符 `'%'` 会**反转**当前的 `result`。

返回最终字符串 `result` 中第 `k` 个字符（下标从 0 开始）。如果 `k` 超出 `result` 的下标索引范围，则返回 `'.'`。

## 输入
字符串 `s`，整数 `k`
## 输出
`result[k]`，若越界则返回 `'.'`
## 约束条件
1. `1 <= s.length <= 10^5`
2. `s` 只包含小写英文字母和特殊字符 `*`、`#` 和 `%`
3. `0 <= k <= 10^15`
4. 处理 `s` 后得到的 `result` 的长度不超过 `10^15`
## 示例1
```
input: s = "a#b%*", k = 1
output: "a"
```
解释：依次得到 `"a"` → `"aa"` → `"aab"` → `"baa"` → `"ba"`，下标 `k = 1` 的字符是 `'a'`
## 示例2
```
input: s = "cd%#*#", k = 3
output: "d"
```
解释：依次得到 `"c"` → `"cd"` → `"dc"` → `"dcdc"` → `"dcd"` → `"dcddcd"`，下标 `k = 3` 的字符是 `'d'`
## 示例3
```
input: s = "z*#", k = 0
output: "."
```
解释：依次得到 `"z"` → `""` → `""`，`result` 为空串，`k = 0` 越界，返回 `'.'`
# 解题思路
这题与 [用特殊操作处理字符串Ⅰ](https://leetcode.cn/problems/process-string-with-special-operations-i/) 的处理规则完全相同，区别在于：Ⅰ 要求返回完整的 `result`，Ⅱ 只要求返回下标 `k` 处的**一个字符**。

## 从模拟出发
Ⅰ 的做法很直接——用列表维护 `result`，从左到右模拟四种操作即可。若把同样思路搬到这里，在 `s` 较短时确实能过：

```python
class Solution:
    def processStr(self, s: str, k: int) -> str:
        res = []
        for c in s:
            if c == '*':
                if res: res.pop()
            elif c == '#': res += res
            elif c == '%': res.reverse()
            else: res.append(c)
        return res[k] if k < len(res) else '.'
```

但本题 `result` 的长度可达 $10^{15}$，`'#'` 每次会把长度翻倍，字符串根本无法在内存中构造出来，模拟这条路走不通。

不过我们并不需要整个 `result`——只要第 `k` 个字符。能否在**不展开字符串**的前提下，把 `k` 映射回某个原始字母呢？

## 正向：只算长度
构造完整字符串做不到，但**追踪长度**可以。从左到右扫一遍 `s`，用变量 `sz` 表示当前 `result` 的长度：

- 普通字母：`sz += 1`
- `'*'`：删除末尾，`sz = max(sz - 1, 0)`
- `'#'`：复制并拼接，`sz += sz`（长度翻倍）
- `'%'`：反转不改变长度，跳过

扫完后 `sz` 就是最终 `result` 的长度。若 `k >= sz`，直接返回 `'.'`。

以示例 1 为例，`"a#b%*"` 的长度变化为 $1 \to 2 \to 3 \to 3 \to 2$，最终 `sz = 2`，`k = 1` 合法。

## 反向：把 k 映射回去
正向知道了「最终有多长」，接下来从**右往左**逆推每个操作对下标 `k` 的影响，同时维护「当前这一步操作之后 `result` 的长度」`sz`（从最终长度出发，逐步还原到操作前的长度）。

逆推时，四种操作的「撤销」规则如下：

**`'*'`（撤销删除）**  
正向删除使长度减 1，撤销则 `sz += 1`。

**`'#'`（撤销复制）**  
正向把长度为 `sz/2` 的串复制成 `sz`，撤销先 `sz //= 2`。复制后 `result = 前半 + 后半`，若 `k` 落在后半段（`k >= sz`），则 `k -= sz`，映射到前半段对应位置。

**`'%'`（撤销反转）**  
反转不改变长度，但下标镜像：`k = sz - k - 1`。

**普通字母**  
正向追加使长度加 1，撤销则 `sz -= 1`。若此时 `k == sz`，说明目标字符就是当前这个字母，直接返回。

以示例 1 演示逆推过程。最终 `sz = 2`，`k = 1`，从右往左处理 `s`：

| 字符 | 操作 | 变化后 `sz` | 变化后 `k` |
| --- | --- | --- | --- |
| `'*'` | 撤销删除 | 3 | 1 |
| `'%'` | 撤销反转 | 3 | $3-1-1=1$ |
| `'b'` | 撤销追加 | 2 | 1（不等于 2，继续） |
| `'#'` | 撤销复制 | 1 | $1-1=0$（$k \ge sz$，减去前半段长度） |
| `'a'` | 撤销追加 | 0 | $k=0=sz$，返回 `'a'` |

示例 2 中 `s = "cd%#*#"`，正向得 `sz = 6`，`k = 3`。逆推时遇到 `'#'` 会将 `k` 从 3 调整为 0，再经 `'%'` 镜像为 1，最终在字母 `'d'` 处命中。

## 复杂度
正向、反向各遍历 `s` 一次，时间复杂度 $O(n)$，空间复杂度 $O(1)$。

## 代码
```python
class Solution:
    def processStr(self, s: str, k: int) -> str:
        sz = 0
        for c in s:
            if c == '*':
                sz = max(sz - 1, 0)
            elif c == '#':
                sz += sz
            elif c == '%': continue
            else: sz += 1
        
        if k >= sz: return '.'

        for c in reversed(s):
            if c == '*': sz += 1
            elif c == '#':
                sz //= 2
                if k >= sz: k -= sz
            elif c == '%': 
                k = sz - k - 1
            else: 
                sz -= 1
                if k == sz: return c
```
