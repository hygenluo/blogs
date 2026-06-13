---
title: 链表最大孪生和
author: Hygen
description: LeetCode-2130
category: Algorithm and Structure
tags: [Solution, 双指针, 链表, DFS]
published: 2026-06-14
---
# 题目描述
[题目链接](https://leetcode.cn/problems/maximum-twin-sum-of-a-linked-list/description/?envType=daily-question&envId=2026-06-14)

在一个大小为`n`且`n`为偶数的链表中，第1个节点和第n个节点为一对孪生节点，第2对与第n-1对为一对孪生节点，依此类推...

**孪生和**为一个节点和它的孪生节点两者之和

给一个长度为`n`的链表`head`，要求返回链表的最大孪生和

## 示例1
> input: head=[5， 4， 2， 1]

> output: 6

解释：5 + 1 = 6， 4 + 2 = 6，最大孪生和为6
## 示例2
> input: head=[4， 2， 2， 3]

> output: 7

解释： 4 + 3 = 7

# 解题思路
如果是在数组上做这道题，那么题目将会非常简单，只需要按顺序求arr[0] + arr[n-1], ... , arr[n // 2 - 1] + arr[n // 2]的最大值即可，但是这题是在链表中，无法随机存取，因此想要从后往前推需要办一些手续。

我们正常遍历链表一定是从左往右，那么如何能让它遍历的顺序从右往左呢？仅需要考虑遍历的话，我们可以使用`DFS`，因为它从左往右走完一遍后必定还要从右往左回来一趟，我们可以考虑仅利用它的后半程，在此期间再运用一个从左往右遍历的指针即可计算孪生和并取最大值。
## 时间复杂度
$O(n)$，n为链表元素个数
## 代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        ans = 0
        l = head

        def dfs(r: Optional[ListNode]):
            if r.next: dfs(r.next)
            nonlocal ans, l
            ans = max(ans, l.val + r.val)
            l = l.next
        
        dfs(head)

        return ans
```