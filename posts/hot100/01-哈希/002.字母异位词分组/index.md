---
title: 字母异位词分组
author: Hygen
description: Leetcode-49
category: hot100
tags: [Solution, 哈希表, 字符串, 排序]
published: 2026-06-15
---
# 题目描述
[题目链接](https://leetcode.cn/problems/group-anagrams/description/?envType=study-plan-v2&envId=top-100-liked)

## 输入
一个字符串数组 `strs`
## 输出
将字母异位词组合在一起，按任意顺序返回结果列表
## 约束条件
1. `1 <= strs.length <= 10^4`
2. `0 <= strs[i].length <= 100`
3. `strs[i]` 仅包含小写字母
## 示例1
```
input: strs = ["eat","tea","tan","ate","nat","bat"]
output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```
解释：`"nat"` 和 `"tan"` 互为字母异位词；`"ate"`、`"eat"`、`"tea"` 互为字母异位词；`"bat"` 单独成组。
## 示例2
```
input: strs = [""]
output: [[""]]
```
## 示例3
```
input: strs = ["a"]
output: [["a"]]
```
# 解题思路
题目要求把**字母异位词**分到同一组。所谓字母异位词，就是两个字符串包含的字符种类和数量完全相同，只是排列顺序不同，比如 `"eat"` 和 `"tea"`。

很容易想到对每个字符串去和其他字符串逐一比较，判断是否为异位词，相同则划入同一组。但外层要对 `n` 个字符串两两配对，整体复杂度约为 $O(n^2 \cdot k)$（`k` 为字符串最大长度），在 `strs.length` 达到 $10^4$ 时不够快，此路不通。

不过我们依旧可以想想：判断两个字符串是否为异位词，有一个很简单的办法——排序后比较。`"eat"` 和 `"tea"` 排序后都是 `"aet"`，而 `"bat"` 排序后是 `"abt"`，彼此不同。这就给了我们一个思路：我们并不需要真的去两两"比较"，只需要为每个字符串生成一个**分组标识**，凡是异位词，标识必须相同；非异位词，标识必须不同。

什么可以作为分组标识？既然异位词排序后结果一样，把每个字符串排序，再拼回字符串，就可以当作 key：

```
"eat" -> "aet"
"tea" -> "aet"
"tan" -> "ant"
```

接下来就是把字符串按标识归类。什么结构适合"按某个 key 分组存放"呢？显然**哈希表**可以，且它的存取都是 $O(1)$ 的。这题每个 key 对应的是一组字符串，value 需要是一个列表，直接用 python 中的 `defaultdict(list)` 即可——遇到新 key 时自动创建空列表，省去手动判断 key 是否存在的步骤。

遍历 `strs` 时，对当前字符串 `s` 计算 `''.join(sorted(s))` 作为 key，把 `s` 本身 `append` 进 `d[key]`。注意存的是**原字符串**，不是排序后的结果，题目要返回的是分组后的原始字符串。

全部遍历结束后，哈希表里每个 value 就是一组异位词。题目要求返回 `List[List[str]]`，而 `d.values()` 拿到的是字典值的视图，用列表推导 `[v for v in d.values()]` 收集成列表返回即可。

时间复杂度 $O(n \cdot k \log k)$，其中 `n` 为字符串个数，`k` 为字符串最大长度（排序开销）；空间复杂度 $O(n \cdot k)$。

## 代码
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = defaultdict(list)
        for s in strs:
            d[''.join(sorted(s))].append(s)
        return [v for v in d.values()]
```
