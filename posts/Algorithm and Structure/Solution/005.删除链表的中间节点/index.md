---
title: 删除链表的中间节点
author: Hygen
description: LeetCode-2095
category: Algorithm and Structure
tags: [Solution, 双指针, 链表]
published: 2026-06-15
---
# 题目描述
[题目链接](https://leetcode.cn/problems/delete-the-middle-node-of-a-linked-list/description/?envType=daily-question&envId=2026-06-15)

给你一个链表的头节点 `head`，**删除**链表的**中间节点**，并返回修改后的链表的头节点 `head`。

长度为 `n` 链表的中间节点是从头数起第 `⌊n / 2⌋` 个节点（下标从 **0** 开始），其中 `⌊x⌋` 表示小于或等于 `x` 的最大整数。

- 对于 `n` = `1`、`2`、`3`、`4` 和 `5` 的情况，中间节点的下标分别是 `0`、`1`、`1`、`2` 和 `2`。

## 约束条件
1. 链表中节点的数目在范围 `[1, 10^5]` 内
2. `1 <= Node.val <= 10^5`
## 示例1
```
input: head = [1,3,4,7,1,2,6]
output: [1,3,4,1,2,6]
```
解释：n = 7，下标为 3 的节点（值为 7）是中间节点，删除后返回新链表。
## 示例2
```
input: head = [1,2,3,4]
output: [1,2,4]
```
解释：n = 4，下标为 2 的节点（值为 3）是中间节点。
## 示例3
```
input: head = [2,1]
output: [2]
```
解释：n = 2，下标为 1 的节点（值为 1）是中间节点，删除后只剩节点 0。
# 解题思路
这题和 [876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/) 很像，区别在于那题只需要**找到**中间节点，而这题需要**删除**它。删除操作需要拿到中间节点的前驱，因此不能直接把慢指针停在中间节点上。

## 两次遍历
最直观的做法：先遍历一遍统计链表长度 `n`，再从头走到第 `⌊n / 2⌋` 个节点的前一个位置，执行 `prev.next = prev.next.next` 即可。为了统一处理删除头节点（`n = 1`）的情况，可以在链表头部加一个哑节点 `dummy`：

```python
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        dummy = ListNode(0, head)
        cur = dummy
        for _ in range(n // 2):
            cur = cur.next
        cur.next = cur.next.next
        return dummy.next
```

时间复杂度 $O(n)$，空间复杂度 $O(1)$，但需要遍历链表两次。

## 快慢指针
能否只遍历一次？可以。快慢指针的经典结论是：快指针每次走两步、慢指针每次走一步，当快指针到达末尾时，慢指针恰好停在中间节点。但删除需要的是**中间节点的前驱**，因此让慢指针比快指针**晚出发一步**——具体做法是用 `dummy` 节点作为慢指针起点，快指针从 `head` 出发，循环条件为 `fast.next and fast.next.next`：

```python
dummy -> 1 -> 3 -> 4 -> 7 -> 1 -> 2 -> 6
slow^   fast^
```

快指针每次跳两步，慢指针每次跟一步。当 `fast.next.next` 不存在时停止，此时 `slow` 指向中间节点的前驱，直接跳过下一个节点即可完成删除。

以 `n = 7` 为例，三轮循环后 `slow` 停在值为 4 的节点，`slow.next` 即为待删的中间节点 7。

## 时间复杂度
$O(n)$，只需遍历链表一次；空间复杂度 $O(1)$。

## 代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        slow = fast = dummy
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        slow.next = slow.next.next
        return dummy.next
```
