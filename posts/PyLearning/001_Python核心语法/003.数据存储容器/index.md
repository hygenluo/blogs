---
title: Python核心语法03-数据存储容器
author: Hygen
description: Python数据存储容器学习
published: 2026-05-24
---

# 1. 数据容器概述

## 1.1. 什么是数据容器

数据容器是一种可以容纳多份数据的数据类型，容纳的每一份数据称之为1个元素，每一个元素都可以是任意类型的数据，如：字符串、数字、布尔等。

在Python中，主要有以下5种数据容器：

- 字符串（str）
- 列表（list）
- 元组（tuple）
- 集合（set）
- 字典（dict）

## 1.2. 为什么需要数据容器

当我们需要存储多个相关数据时，如果使用单个变量会非常繁琐：

```python
# 单个变量存储多个成绩
score1 = 695
score2 = 558
score3 = 622
score4 = 589
score5 = 645
# ... 更多变量
```

使用数据容器可以让代码更加简洁优雅：

```python
# 列表存储多个成绩
score_list = [695, 558, 622, 589, 645, 607, 577, 552, 475, 602]
```

## 1.3. 数据容器特性对比

| 特性 | 字符串（str） | 列表（list） | 元组（tuple） | 集合（set） | 字典（dict） |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 有序性 | 有序 | 有序 | 有序 | 无序 | 有序(3.7+) |
| 重复元素 | 允许 | 允许 | 允许 | 不允许 | key不允许 |
| 可变性 | 不可变 | 可变 | 不可变 | 可变 | 可变 |
| 索引访问 | 支持 | 支持 | 支持 | 不支持 | 不支持 |
| 切片操作 | 支持 | 支持 | 支持 | 不支持 | 不支持 |
| 使用场景 | 文本处理 | 有序可重复数据集合 | 固定数据记录 | 去重数据集合 | 键值对 |

# 2. 列表（list）

## 2.1. 列表基础

### 2.1.1. 列表介绍

列表是数据容器中的一类，是一次性可以存储多个数据（元素）的。

**特点：**
- 可以存储不同类型的元素
- 元素有序
- 元素可以重复
- 元素可以修改

### 2.1.2. 列表定义

```python
# 定义列表
列表名称 = [元素1, 元素2, 元素3, 元素4, 元素5...]

# 示例
score_list = [695, 558, 622, 589, 645]
mixed_list = [100, "Python", True, 3.14]
```

### 2.1.3. 列表索引

列表中的每一个元素都有其对应的下标（索引），通过元素对应的索引，就可以获取到对应的元素。

- 从前向后（正向索引），下标从0开始
- 从后向前（反向索引），下标从-1开始

```python
s = [54, 15, 75, 108, 23, 78, 75]

# 正向索引
print(s[0])  # 输出: 54
print(s[1])  # 输出: 15
print(s[6])  # 输出: 75

# 反向索引
print(s[-1])  # 输出: 75
print(s[-2])  # 输出: 78
print(s[-7])  # 输出: 54
```

**注意：** 如果指定的索引值超出范围，将会报错。

### 2.1.4. 列表元素的基本操作

```python
# 查看元素
print(score_list[0])

# 修改元素
score_list[0] = 700
print(score_list)  # 输出: [700, 558, 622, 589, 645]

# 删除元素
del score_list[3]
print(score_list)  # 输出: [700, 558, 622, 645]
```

## 2.2. 列表切片

### 2.2.1. 切片介绍

切片是指对操作的数据截取其中一部分的操作。列表、字符串、元组都支持切片操作。

### 2.2.2. 切片语法

```python
序列数据[开始索引:结束索引:步长]
```

**特点：**
- 不包含结束索引位置对应的元素
- 开始索引未指定默认为0
- 结束索引未指定默认为列表长度（直到列表末尾）
- 步长未指定默认为1
- 索引采用正向、反向索引都可以
- 步长是选取间隔

### 2.2.3. 切片示例

```python
s = ["A", "B", "C", "E", "D", "E", "G"]

# 基本切片
print(s[0:5:1])  # 输出: ["A", "B", "C", "E", "D"]
print(s[0:5:2])  # 输出: ["A", "C", "D"]

# 省略步长
print(s[0:5])  # 输出: ["A", "B", "C", "E", "D"]

# 省略开始索引
print(s[:5])  # 输出: ["A", "B", "C", "E", "D"]

# 省略结束索引
print(s[2:])  # 输出: ["C", "E", "D", "E", "G"]

# 反向切片
print(s[::-1])  # 输出: ["G", "E", "D", "E", "C", "B", "A"]
```

## 2.3. 列表常用方法

列表的常用方法就是指列表这种数据类型内置的常见功能（添加元素、删除元素、排序等）。

| 方法 | 作用 | 样例 |
| :---: | :--- | :--- |
| `append()` | 在列表的尾部追加元素 | `s.append(10086)` |
| `insert()` | 在指定索引之前，插入该元素 | `s.insert(0, 92)` |
| `remove()` | 移除列表中第一个匹配到的值 | `s.remove(75)` |
| `pop()` | 删除列表中指定索引位置的元素（如果未指定索引，默认删最后一个） | `s.pop(2)` / `s.pop()` |
| `sort()` | 对列表进行排序（列表元素的数据类型一致，才可以进行排序） | `s.sort()` |
| `reverse()` | 反转列表元素 | `s.reverse()` |

### 2.3.1. 添加元素

```python
# 尾部追加元素
score_list = [695, 558, 622]
score_list.append(589)
print(score_list)  # 输出: [695, 558, 622, 589]

# 指定位置插入元素
score_list.insert(1, 645)
print(score_list)  # 输出: [695, 645, 558, 622, 589]
```

### 2.3.2. 删除元素

```python
# 删除指定值的第一个元素
score_list = [695, 558, 622, 558, 645]
score_list.remove(558)
print(score_list)  # 输出: [695, 622, 558, 645]

# 删除指定索引的元素
score_list.pop(2)
print(score_list)  # 输出: [695, 622, 645]

# 删除最后一个元素
score_list.pop()
print(score_list)  # 输出: [695, 622]
```

### 2.3.3. 排序与反转

```python
# 排序
num_list = [54, 15, 75, 108, 23, 78]
num_list.sort()
print(num_list)  # 输出: [15, 23, 54, 75, 78, 108]

# 降序排序
num_list.sort(reverse=True)
print(num_list)  # 输出: [108, 78, 75, 54, 23, 15]

# 反转列表
num_list.reverse()
print(num_list)  # 输出: [15, 23, 54, 75, 78, 108]
```

## 2.4. 列表常用操作

### 2.4.1. 列表合并

```python
# 使用+运算符合并
num_list1 = [19, 23, 54, 64, 875]
num_list2 = [55, 80, 72, 35, 60]
merged_list = num_list1 + num_list2
print(merged_list)  # 输出: [19, 23, 54, 64, 875, 55, 80, 72, 35, 60]

# 使用*解包操作
merged_list = [*num_list1, *num_list2]
print(merged_list)  # 输出: [19, 23, 54, 64, 875, 55, 80, 72, 35, 60]
```

### 2.4.2. 元素存在性判断

```python
num_list = [19, 23, 54, 64, 875]
print(23 in num_list)  # 输出: True
print(100 in num_list)  # 输出: False
print(100 not in num_list)  # 输出: True
```

### 2.4.3. 统计函数

Python中提供了一些用于数据统计的内置函数：

| 函数 | 作用 |
| :---: | :--- |
| `min()` | 获取最小值 |
| `max()` | 获取最大值 |
| `sum()` | 求和 |
| `len()` | 获取元素的个数 |

```python
score_list = [695, 558, 622, 589, 645]
print(min(score_list))  # 输出: 558
print(max(score_list))  # 输出: 695
print(sum(score_list))  # 输出: 3109
print(len(score_list))  # 输出: 5
```

## 2.5. 列表推导式

### 2.5.1. 什么是列表推导式

列表推导式就是按照一定规则快速生成一个列表的方法。

### 2.5.2. 列表推导式语法

```python
# 格式1：基本格式
列表名称 = [要插入列表的数据 for i in 列表]

# 格式2：带条件格式
列表名称 = [要插入列表的数据 for i in 列表 if 条件]
```

### 2.5.3. 列表推导式示例

```python
# 生成1-20的平方列表
square_list = [i ** 2 for i in range(1, 21)]
print(square_list)

# 从数字列表中提取所有偶数，并计算其平方
num_list = [19, 23, 54, 64, 87, 20, 109, 232, 123, 43, 26, 55, 72]
even_square_list = [i ** 2 for i in num_list if i % 2 == 0]
print(even_square_list)
```

## 2.6. 案例实操

### 2.6.1. 案例1：成绩统计

需求：将用户输入的10个数字，存储到一个列表中，并将列表中的数字进行排序，输出其中的最小值、最大值和平均值。

```python
# 存储用户输入的成绩
scores = []

# 循环输入10个成绩
for i in range(10):
    score = int(input(f"请输入第{i+1}个成绩: "))
    scores.append(score)

# 排序
scores.sort()
print("排序后的成绩:", scores)

# 计算统计值
min_score = min(scores)
max_score = max(scores)
avg_score = sum(scores) / len(scores)

print(f"最小值: {min_score}")
print(f"最大值: {max_score}")
print(f"平均值: {avg_score:.2f}")
```

### 2.6.2. 案例2：列表合并与去重

需求：合并两个列表中的元素，并对合并的结果进行去重处理。

```python
# 定义列表
num_list1 = [19, 23, 54, 64, 875, 20, 109, 232, 123, 54]
num_list2 = [55, 80, 72, 35, 60, 123, 54, 29, 91]

# 合并列表
merged_list = num_list1 + num_list2

# 去重
unique_list = []
for num in merged_list:
    if num not in unique_list:
        unique_list.append(num)

print("合并后的列表:", merged_list)
print("去重后的列表:", unique_list)
```

# 3. 字符串（str）

## 3.1. 字符串基础

### 3.1.1. 字符串介绍

字符串是字符的容器，一个字符串中可以存放任意数量的字符。

**特点：**
- 不可变性（无法修改）
- 有序性
- 可迭代性

### 3.1.2. 字符串定义

```python
# 单引号定义
s1 = 'Python'

# 双引号定义
s2 = "Python"

# 三引号定义（支持多行）
s3 = """Python
是一门
编程语言"""
```

### 3.1.3. 字符串索引

字符串中的每一个字符元素都有其对应的下标（索引），通过元素对应的索引，就可以获取到对应的元素。

```python
s = "Python"

# 正向索引
print(s[0])  # 输出: P
print(s[1])  # 输出: y
print(s[5])  # 输出: n

# 反向索引
print(s[-1])  # 输出: n
print(s[-2])  # 输出: o
print(s[-6])  # 输出: P
```

## 3.2. 字符串切片

字符串也支持切片操作，语法与列表切片相同。

```python
s = "Python"

# 基本切片
print(s[0:5:1])  # 输出: Pytho
print(s[0:5:2])  # 输出: Pto

# 反向切片
print(s[::-1])  # 输出: nohtyP
```

## 3.3. 字符串常用方法

| 方法 | 作用 | 样例 |
| :---: | :--- | :--- |
| `find()` | 在字符串中查找子串，返回第一次出现的索引位置，找不到返回-1 | `s.find('Python')` |
| `count()` | 统计子串在字符串中出现的次数 | `s.count('H')` |
| `upper()` | 将字符串中的所有字母转换为大写 | `s.upper()` |
| `lower()` | 将字符串中的所有字母转换为小写 | `s.lower()` |
| `split()` | 将字符串按指定分隔符分割成列表 | `s.split(' ')` |
| `strip()` | 去除字符串两端的空白字符或指定字符 | `s.strip()` / `s.strip('*')` |
| `replace()` | 将字符串中的指定子串替换为新的子串 | `s.replace('H','C')` |
| `startswith()` | 检查字符串是否以指定子串开头，返回布尔值 | `s.startswith('P')` |

### 3.3.1. 查找与统计

```python
s = "Hello World, Hello Python"

# 查找子串
print(s.find("Hello"))  # 输出: 0
print(s.find("Python"))  # 输出: 13
print(s.find("Java"))  # 输出: -1

# 统计子串出现次数
print(s.count("Hello"))  # 输出: 2
print(s.count("o"))  # 输出: 4
```

### 3.3.2. 大小写转换

```python
s = "Hello World"

# 转换为大写
print(s.upper())  # 输出: HELLO WORLD

# 转换为小写
print(s.lower())  # 输出: hello world
```

### 3.3.3. 分割与连接

```python
# 分割字符串
s = "apple,banana,orange"
fruit_list = s.split(",")
print(fruit_list)  # 输出: ['apple', 'banana', 'orange']

# 连接字符串
fruit_str = "-".join(fruit_list)
print(fruit_str)  # 输出: apple-banana-orange
```

### 3.3.4. 去除空白字符

```python
s = "   Hello World   "
print(s.strip())  # 输出: Hello World
print(s.lstrip())  # 输出: Hello World   
print(s.rstrip())  # 输出:    Hello World
```

### 3.3.5. 替换子串

```python
s = "Hello World"
new_s = s.replace("World", "Python")
print(new_s)  # 输出: Hello Python
```

## 3.4. 字符串常用操作

### 3.4.1. 子串存在性判断

```python
s = "Hello World"
print("Hello" in s)  # 输出: True
print("Python" in s)  # 输出: False
```

### 3.4.2. 字符串拼接

```python
# 使用+运算符拼接
s1 = "Hello"
s2 = "World"
s3 = s1 + " " + s2
print(s3)  # 输出: Hello World

# 使用f-string格式化
name = "Tom"
age = 18
s = f"我叫{name}，今年{age}岁"
print(s)  # 输出: 我叫Tom，今年18岁
```

## 3.5. 案例实操

### 3.5.1. 案例1：邮箱格式验证

需求：用户输入一个邮箱，验证邮箱格式是否正确（包含一个@和至少一个.），如果输入正确，输出"邮箱格式正确"，否则输出"邮箱格式错误"。

```python
email = input("请输入邮箱地址: ")

if "@" in email and "." in email:
    print("邮箱格式正确")
else:
    print("邮箱格式错误")
```

### 3.5.2. 案例2：判断回文串

需求：输入一个字符串，判断该字符串是否是回文（两边对称）。

```python
s = input("请输入一个字符串: ")

# 去除空格并转换为小写
s = s.replace(" ", "").lower()

# 判断是否是回文
if s == s[::-1]:
    print("是回文串")
else:
    print("不是回文串")
```

# 4. 元组（tuple）

## 4.1. 元组基础

### 4.1.1. 元组介绍

元组是不可变的序列，类似于列表，但创建后不能修改。

**特点：**
- 可以存储不同类型的元素
- 元素可以重复
- 有序
- 不可以修改（只读）
- 支持索引访问和切片

### 4.1.2. 元组定义

```python
# 定义元组
元组名称 = (元素1, 元素2, ...)

# 定义空元组
元组名称 = ()
元组名称 = tuple()

# 示例
t1 = (5, 7, 9, 1, 2, 3)
t2 = ("Python", 3.14, True)
t3 = ()
t4 = tuple()
```

**注意：** 定义单元素元组时，需要在结尾加上逗号。

```python
# 错误写法（这不是元组，而是一个整数）
t = (5)
print(type(t))  # 输出: <class 'int'>

# 正确写法
t = (5,)
print(type(t))  # 输出: <class 'tuple'>
```

## 4.2. 元组常用方法

由于元组是不可变的，所以它的方法主要是查询方法：

| 方法 | 作用 |
| :---: | :--- |
| `count()` | 统计某元素在元组中出现的次数 |
| `index()` | 查找某个元素在元组中的索引位置（第一次出现的位置） |

```python
t1 = (5, 7, 9, 1, 2, 3, 10, 6, 4, 8, 12, 7, 5)

# 统计元素出现次数
print(t1.count(7))  # 输出: 2
print(t1.count(5))  # 输出: 2

# 查找元素索引
print(t1.index(9))  # 输出: 2
print(t1.index(12))  # 输出: 10
```

## 4.3. 组包与解包

### 4.3.1. 什么是组包与解包

- **组包（Packing）**：将多个值合并到一个容器（元组、列表）中。
- **解包（Unpacking）**：将容器（元组、列表）解开成独立的元素，分别赋值给多个变量。

### 4.3.2. 组包示例

```python
# 组包
t1 = (5, 7, 9, 1)
t2 = 5, 7, 9, 1  # 省略括号也是元组

print(type(t1))  # 输出: <class 'tuple'>
print(type(t2))  # 输出: <class 'tuple'>
```

### 4.3.3. 基础解包

```python
t1 = (5, 7, 9, 1)

# 基础解包
a, b, c, d = t1
print(a)  # 输出: 5
print(b)  # 输出: 7
print(c)  # 输出: 9
print(d)  # 输出: 1
```

### 4.3.4. 扩展解包

在元组解包时，`*` 表示收集剩余的所有元素，允许我们处理不确定数量的元素（生成列表）。

```python
t2 = (5, 7, 9, 1)

# 扩展解包
x, *y, z = t2
print(x)  # 输出: 5
print(y)  # 输出: [7, 9]
print(z)  # 输出: 1

s, *o = t2
print(s)  # 输出: 5
print(o)  # 输出: [7, 9, 1]

*o, e = t2
print(o)  # 输出: [5, 7, 9]
print(e)  # 输出: 1
```

## 4.4. 案例实操

### 4.4.1. 案例1：变量交换

需求：现有两个变量，分别为：a = 10，b = 20，现需要将这两个变量值交换，然后输出到控制台。

```python
a = 10
b = 20

# 传统方法
temp = a
a = b
b = temp
print(a, b)  # 输出: 20 10

# 使用元组解包
a = 10
b = 20
a, b = b, a
print(a, b)  # 输出: 20 10
```

### 4.4.2. 案例2：多变量交换

需求：现有三个变量，分别为：a = 100，b = 200，c = 300，现需要将这三个变量值进行交换，将a,b,c的值分别赋值给c,a,b，并将其输出到控制台。

```python
a = 100
b = 200
c = 300

# 使用元组解包
a, b, c = b, c, a
print(a, b, c)  # 输出: 200 300 100
```

# 5. 集合（set）

## 5.1. 集合基础

### 5.1.1. 集合介绍

集合（set）是一种无序的、不可重复、可修改的数据容器。

**特点：**
- 无序
- 不可重复
- 可修改
- 不支持下标索引访问

### 5.1.2. 集合定义

```python
# 定义集合
s1 = {"C", "D", "X", "T", "O", "U"}

# 定义空集合
s2 = set()

# 错误写法（这是空字典，不是空集合）
s3 = {}
```

**注意：** 空集合的定义不可以使用`{}`，`{}`表示的是空字典。

## 5.2. 集合常用操作

| 操作 | 含义 | 样例 |
| :---: | :--- | :--- |
| `add(..)` | 添加元素到集合中 | `s1.add('t')` |
| `remove(..)` | 移除集合中的指定元素（指定元素不存在将报错） | `s1.remove('t')` |
| `pop()` | 随机删除集合中的元素并返回 | `e = s1.pop()` |
| `clear()` | 清空集合 | `s1.clear()` |
| `difference()` | 求取两个集合的差集（包含在第一个集合但不包含在第二个集合的元素） | `s1.difference(s2)` |
| `union()` | 求取两个集合的并集 | `s1.union(s2)` |
| `intersection()` | 求取两个集合的交集 | `s1.intersection(s2)` |

### 5.2.1. 添加元素

```python
s = {"A", "B", "C"}
s.add("D")
print(s)  # 输出: {'A', 'B', 'C', 'D'}
```

### 5.2.2. 删除元素

```python
s = {"A", "B", "C", "D"}

# 删除指定元素
s.remove("B")
print(s)  # 输出: {'A', 'C', 'D'}

# 随机删除一个元素
e = s.pop()
print(e)
print(s)

# 清空集合
s.clear()
print(s)  # 输出: set()
```

### 5.2.3. 集合运算

```python
s1 = {1, 2, 3, 4, 5}
s2 = {4, 5, 6, 7, 8}

# 交集
print(s1.intersection(s2))  # 输出: {4, 5}
print(s1 & s2)  # 输出: {4, 5}

# 并集
print(s1.union(s2))  # 输出: {1, 2, 3, 4, 5, 6, 7, 8}
print(s1 | s2)  # 输出: {1, 2, 3, 4, 5, 6, 7, 8}

# 差集
print(s1.difference(s2))  # 输出: {1, 2, 3}
print(s1 - s2)  # 输出: {1, 2, 3}
```

## 5.3. 集合推导式

集合也支持推导式写法：

```python
# 基本格式
变量名称 = {i表达式 for i in 列表}

# 带条件格式
变量名称 = {i表达式 for i in 列表 if 条件}
```

```python
# 生成1-10的平方集合
square_set = {i ** 2 for i in range(1, 11)}
print(square_set)

# 从列表中提取所有偶数
num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_set = {i for i in num_list if i % 2 == 0}
print(even_set)
```

## 5.4. 案例实操

### 5.4.1. 案例1：选课统计

需求：根据提供的班级学生的选课情况，完成如下需求：
1. 找出同时选修了法语和艺术的学生
2. 找出同时选修了所有四门课程的学生
3. 找出选修了足球，但是没有选修篮球的学生
4. 统计每一个学生选修的课程数量

```python
# 选修足球学生名单
football_set = {"王林", "曾牛", "徐立国", "遁天", "天运子", "韩立", "厉飞雨", "乌丑", "紫灵"}
# 选修篮球学生名单
basketball_set = {"张铁", "墨居仁","王林", "姜老道", "曾牛", "王蝉", "韩立", "天运子", "李化元", "厉飞雨", "云露"}
# 选修法语学生名单
french_set = {"许木", "王卓", "十三", "虎咆", "姜老道", "天运子",  "红蝶", "厉飞雨", "韩立", "曾牛"}
# 选修艺术学生名单
art_set = {"遁天", "天运子", "韩立", "虎咆", "姜老道", "紫灵"}

# 1. 同时选修了法语和艺术的学生
both_french_art = french_set & art_set
print("同时选修了法语和艺术的学生:", both_french_art)

# 2. 同时选修了所有四门课程的学生
all_courses = football_set & basketball_set & french_set & art_set
print("同时选修了所有四门课程的学生:", all_courses)

# 3. 选修了足球，但是没有选修篮球的学生
football_not_basketball = football_set - basketball_set
print("选修了足球，但是没有选修篮球的学生:", football_not_basketball)

# 4. 统计每一个学生选修的课程数量
all_students = football_set | basketball_set | french_set | art_set
for student in all_students:
    count = 0
    if student in football_set:
        count += 1
    if student in basketball_set:
        count += 1
    if student in french_set:
        count += 1
    if student in art_set:
        count += 1
    print(f"{student} 选修了 {count} 门课程")
```

# 6. 字典（dict）

## 6.1. 字典基础

### 6.1.1. 字典介绍

Python中的字典（dict），里面存储的是键值对（key: value）类型的数据，可以根据键（key）找到对应的值（value）。

**特点：**
- 键值对（key: value）存储
- 键（key）不能重复
- 可修改
- 没有索引下标，不能根据索引获取值，只可以根据key获取value

### 6.1.2. 字典定义

```python
# 定义字典
字典名称 = {key: value, key: value, key: value …}

# 定义空字典
字典名称 = {}
字典名称 = dict()

# 示例
dict1 = {"王林": 670, "韩立": 556, "李慕婉": 582, "紫灵": 435, "许立国": 608}
dict2 = {}
dict3 = dict()
```

**注意：**
- 字典（dict）中的value可以是任何类型的数据
- 而key不能为可变类型（如：不能为列表list、集合set、字典dict）
- 字典内的key不允许重复，如果重复定义，后面的覆盖前面的

### 6.1.3. 根据key获取value

```python
dict1 = {"王林": 670, "韩立": 556, "李慕婉": 582}

# 根据key获取value
score = dict1["王林"]
print(score)  # 输出: 670

# 如果key不存在，会报错
# score = dict1["张三"]  # KeyError: '张三'
```

## 6.2. 字典常用操作

| 类型 | 操作 | 含义 | 样例 |
| :---: | :---: | :--- | :--- |
| 添加 | `字典名称[key] = value` | 往指定字典中添加key-value键值对 | `dict1["涛哥"] = 688` |
| 删除 | `字典名称.pop(key)` | 删除字典中指定的key，并返回该key对应的value | `score = dict1.pop("涛哥")` |
| | `del 字典名称[key]` | 删除字典中指定的键值对 | `del dict1["涛哥"]` |
| 修改 | `字典名称[key] = value` | 修改字典中指定的key对应的值 | `dict1["小智"] = 658` |
| 查询 | `字典名称[key]` | 根据key获取value | `dict1["涛哥"]` |
| | `字典名称.get(key)` | 根据key获取value | `dict1.get("涛哥")` |
| | `字典名称.keys()` | 获取所有的key | `dict1.keys()` |
| | `字典名称.values()` | 获取所有的value | `dict1.values()` |
| | `字典名称.items()` | 获取所有的key-value键值对 | `dict1.items()` |

### 6.2.1. 添加与修改

```python
dict1 = {"小智": 675, "李思": 608, "李琪": 478}

# 添加元素
dict1["小黑"] = 545
print(dict1)  # 输出: {'小智': 675, '李思': 608, '李琪': 478, '小黑': 545}

# 修改元素
dict1["李思"] = 650
print(dict1)  # 输出: {'小智': 675, '李思': 650, '李琪': 478, '小黑': 545}
```

### 6.2.2. 删除元素

```python
dict1 = {"小智": 675, "李思": 608, "李琪": 478, "小黑": 545}

# 删除指定key
del dict1["李琪"]
print(dict1)  # 输出: {'小智': 675, '李思': 608, '小黑': 545}

# 删除指定key并返回value
score = dict1.pop("小黑")
print(score)  # 输出: 545
print(dict1)  # 输出: {'小智': 675, '李思': 608}
```

### 6.2.3. 查询元素

```python
dict1 = {"小智": 675, "李思": 608, "李琪": 478}

# 根据key获取value
print(dict1["小智"])  # 输出: 675

# 使用get方法获取value（key不存在时返回None）
print(dict1.get("李思"))  # 输出: 608
print(dict1.get("张三"))  # 输出: None

# 获取所有key
print(dict1.keys())  # 输出: dict_keys(['小智', '李思', '李琪'])

# 获取所有value
print(dict1.values())  # 输出: dict_values([675, 608, 478])

# 获取所有键值对
print(dict1.items())  # 输出: dict_items([('小智', 675), ('李思', 608), ('李琪', 478)])
```

## 6.3. 字典遍历

字典支持for循环遍历：

```python
dict1 = {"小智": 675, "李思": 608, "李琪": 478}

# 遍历key
for key in dict1:
    print(key, dict1[key])

# 遍历key（推荐写法）
for key in dict1.keys():
    print(key, dict1[key])

# 遍历value
for value in dict1.values():
    print(value)

# 遍历键值对
for key, value in dict1.items():
    print(key, value)
```

## 6.4. 案例实操

### 6.4.1. 案例1：购物车管理系统

需求：开发一个购物车管理系统，实现商品信息的添加、修改、删除、查询功能。系统使用字典结构存储商品数据，通过控制台菜单与用户交互。

```python
# 购物车字典，key是商品名称，value是包含价格和数量的字典
shopping_cart = {}

while True:
    print("=" * 30)
    print("欢迎使用购物车系统")
    print("1. 添加购物车")
    print("2. 修改购物车")
    print("3. 删除购物车")
    print("4. 查询购物车")
    print("5. 退出购物车")
    print("=" * 30)
    
    choice = input("请选择要执行的操作(1-5): ")
    
    if choice == "1":
        # 添加购物车
        name = input("请输入商品名称: ")
        price = float(input("请输入商品价格: "))
        num = int(input("请输入商品数量: "))
        shopping_cart[name] = {"price": price, "num": num}
        print("商品添加成功！")
    
    elif choice == "2":
        # 修改购物车
        name = input("请输入要修改的商品名称: ")
        if name in shopping_cart:
            price = float(input("请输入新的商品价格: "))
            num = int(input("请输入新的商品数量: "))
            shopping_cart[name] = {"price": price, "num": num}
            print("商品修改成功！")
        else:
            print("商品不存在！")
    
    elif choice == "3":
        # 删除购物车
        name = input("请输入要删除的商品名称: ")
        if name in shopping_cart:
            del shopping_cart[name]
            print("商品删除成功！")
        else:
            print("商品不存在！")
    
    elif choice == "4":
        # 查询购物车
        if not shopping_cart:
            print("购物车为空！")
        else:
            print("购物车商品列表:")
            total_price = 0
            for name, info in shopping_cart.items():
                price = info["price"]
                num = info["num"]
                item_total = price * num
                total_price += item_total
                print(f"商品名称: {name}, 商品价格: {price}, 商品数量: {num}, 小计: {item_total}")
            print(f"购物车总价: {total_price}")
    
    elif choice == "5":
        # 退出购物车
        print("感谢使用购物车系统，再见！")
        break
    
    else:
        print("输入错误，请重新选择！")
```

# 7. 本章总结

## 7.1. 数据容器总结

1. **列表（list）**：有序、可重复、可修改，支持索引和切片，适用于存储有序可重复的数据集合。
2. **字符串（str）**：有序、可重复、不可修改，支持索引和切片，适用于文本处理。
3. **元组（tuple）**：有序、可重复、不可修改，支持索引和切片，适用于存储固定数据记录。
4. **集合（set）**：无序、不可重复、可修改，不支持索引和切片，适用于去重数据集合。
5. **字典（dict）**：有序（3.7+）、key不可重复、可修改，不支持索引和切片，适用于存储键值对数据。

## 7.2. 常用操作总结

1. **索引访问**：列表、字符串、元组支持通过索引访问元素，正向索引从0开始，反向索引从-1开始。
2. **切片操作**：列表、字符串、元组支持切片操作，语法为`[开始索引:结束索引:步长]`。
3. **添加元素**：列表使用`append()`和`insert()`，集合使用`add()`，字典使用`字典[key] = value`。
4. **删除元素**：列表使用`remove()`和`pop()`，集合使用`remove()`和`pop()`，字典使用`del`和`pop()`。
5. **遍历**：所有数据容器都支持for循环遍历，字典可以通过`keys()`、`values()`和`items()`遍历。
6. **推导式**：列表、集合、字典都支持推导式，可以快速生成数据容器。

## 7.3. 选择建议

- 如果需要存储有序的、可修改的数据，使用**列表**。
- 如果需要处理文本数据，使用**字符串**。
- 如果需要存储固定不变的数据，使用**元组**。
- 如果需要对数据进行去重处理，使用**集合**。
- 如果需要通过键快速查找值，使用**字典**。