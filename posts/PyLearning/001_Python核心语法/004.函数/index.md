---
title: Python核心语法04-函数
author: Hygen
description: Python函数学习
published: 2026-05-24
---

# 1. 函数基础

## 1.1. 函数定义

### 1.1.1. 什么是函数

函数是组织好的、可重复使用的、用来实现特定功能的代码片段。

在前面的学习中，我们已经使用过一些 Python 内置函数，例如：

| 函数 | 作用 |
| :---: | :--- |
| `input()` | 获取键盘输入 |
| `print()` | 输出内容到控制台 |
| `max()` | 获取最大值 |
| `min()` | 获取最小值 |
| `len()` | 获取长度 |
| `sum()` | 求和 |

这些函数都是 Python 提前定义好的，可以直接调用、重复使用，并完成特定功能。

### 1.1.2. 函数定义与调用

定义函数使用 `def` 关键字，基本格式如下：

```python
# 定义函数
def 函数名(参数列表):
    函数体
    ......
    return 返回值

# 调用函数
函数名(参数)
```

其中，参数列表和返回值不是必须的，需要根据实际需求决定。

### 1.1.3. 简单示例

```python
# 定义函数
def out_line():
    print("---------------------------")

# 调用函数
out_line()
```

输出结果：

```python
---------------------------
```

### 1.1.4. 函数使用注意事项

1. 函数必须先定义，再调用。
2. 函数定义时，函数体代码不会立即执行。
3. 只有调用函数时，函数体中的代码才会执行。
4. Python 通过缩进表示函数体的归属关系。

## 1.2. 函数的参数与返回值

### 1.2.1. 参数与返回值的作用

在定义函数时，可以根据业务需求设置参数和返回值。

- 参数：函数运行时需要接收的数据。
- 返回值：函数运行后返回给调用者的结果。

基本格式如下：

```python
def 函数名(参数列表):
    函数体
    return 返回值
```

### 1.2.2. 单个参数与返回值

示例：计算圆的面积。

```python
def circle_area(r):
    area = 3.14 * r * r
    return area

c_area = circle_area(10)
print(c_area)
```

输出结果：

```python
314.0
```

说明：

- `r` 是函数的参数，表示圆的半径。
- `return area` 表示将计算结果返回。
- `print(c_area)` 才会把结果输出到控制台。

### 1.2.3. 多个参数与返回值

示例：计算长方形面积。

```python
def rectangle_area(l, w):
    area = l * w
    return area

r_area = rectangle_area(20, 10)
print(r_area)
```

输出结果：

```python
200
```

注意：多个参数之间使用英文逗号 `,` 分隔。

### 1.2.4. 形参与实参

| 名称 | 含义 |
| :---: | :--- |
| 形参 | 函数定义时括号中的参数，只能在函数内部使用 |
| 实参 | 函数调用时实际传入的数据 |

示例：

```python
def circle_area(r):
    return 3.14 * r * r

print(circle_area(10))
```

其中：

- `r` 是形参。
- `10` 是实参。

### 1.2.5. 多个返回值

Python 函数可以返回多个值，多个返回值会被封装成元组。

```python
def calc(a, b):
    return a + b, a - b, a * b

result = calc(10, 5)
print(result)
```

输出结果：

```python
(15, 5, 50)
```

也可以通过元组解包接收多个返回值：

```python
x, y, z = calc(10, 5)
print(x)
print(y)
print(z)
```

输出结果：

```python
15
5
50
```

### 1.2.6. return 与 print 的区别

`return` 只是把结果返回给函数调用处，并不会直接输出内容。

如果想在控制台看到结果，需要结合 `print()` 使用。

```python
def add(a, b):
    return a + b

result = add(10, 20)
print(result)
```

输出结果：

```python
30
```

## 1.3. 函数说明文档

### 1.3.1. 什么是函数说明文档

函数说明文档，也叫 Docstring，是写在函数开头、用三引号包裹的字符串。

它通常用于说明：

1. 函数的功能。
2. 参数的含义。
3. 返回值的含义。

良好的函数说明文档可以让代码更容易理解、使用和维护。

### 1.3.2. 函数说明文档示例

```python
def circle_area_len(r):
    """
    该函数用于根据圆的半径，计算圆的面积和圆的周长
    :param r: 圆的半径
    :return: 圆的面积，圆的周长
    """
    return 3.14 * r * r, 2 * 3.14 * r

al = circle_area_len(10)
print(al)
```

输出结果：

```python
(314.0, 62.800000000000004)
```

### 1.3.3. 查看函数说明文档

可以使用 `help()` 函数查看函数说明文档：

```python
help(circle_area_len)
```

在开发工具中，也可以将鼠标悬浮在函数上查看文档说明。

### 1.3.4. 函数嵌套调用

函数嵌套调用指的是：在一个函数中调用另一个函数。

函数调用遵循栈结构，特点是：

> 最后被调用的函数最先返回。

也就是 LIFO：

> Last In First Out，后进先出。

示例代码：

```python
def function_a():
    print("a ... before")
    function_b()
    print("a ... after")

def function_b():
    print("b ... before")
    function_c()
    print("b ... after")

def function_c():
    print("c ...")

function_a()
```

输出结果：

```python
a ... before
b ... before
c ...
b ... after
a ... after
```

调用过程如下：

```python
function_a() 开始
  │
  ├─ 打印 "a ... before"
  │
  ├─ 调用 function_b()
  │   │
  │   ├─ 打印 "b ... before"
  │   │
  │   ├─ 调用 function_c()
  │   │   │
  │   │   └─ 打印 "c ..."
  │   │
  │   ├─ 打印 "b ... after"
  │   │
  │   └─ function_b() 返回
  │
  ├─ 打印 "a ... after"
  │
  └─ function_a() 返回
```

## 1.4. 案例实操

### 1.4.1. 案例1：计算三角形面积

需求：定义一个函数，根据传入的底和高计算三角形面积。

三角形面积公式：

```python
面积 = 底 * 高 / 2
```

代码实现：

```python
def triangle_area(b, h):
    """
    该函数用于根据底和高计算三角形的面积
    :param b: 三角形的底
    :param h: 三角形的高
    :return: 三角形的面积
    """
    return (b * h) / 2

print(triangle_area(10, 20))
```

输出结果：

```python
100.0
```

### 1.4.2. 案例2：统计元音字母个数

需求：定义一个函数，计算传入字符串中元音字母的个数。

元音字母包括：

```python
aeiouAEIOU
```

代码实现：

```python
def count_vowel(s):
    """
    该函数用于统计指定字符串中元音字母的个数
    :param s: 需要统计的字符串
    :return: 元音字母的个数
    """
    count = 0
    for i in s:
        if i in "aeiouAEIOU":
            count += 1
    return count

print(count_vowel("hello world"))
```

输出结果：

```python
3
```

### 1.4.3. 案例3：统计成绩信息

需求：定义一个函数，计算传入成绩列表中的最高分、最低分和平均分，并返回。

```python
def calc_score(scores):
    """
    该函数用于计算班级学员成绩列表中的最高分、最低分和平均分
    :param scores: 成绩列表
    :return: 最高分、最低分、平均分
    """
    max_score = max(scores)
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)
    avg_score = round(avg_score, 1)
    return max_score, min_score, avg_score

print(calc_score([672, 435, 544, 567, 705, 668, 634, 527, 540, 525, 645, 621, 589, 612]))
```

输出结果：

```python
(705, 435, 591.7)
```

### 1.4.4. 练习1：根据分数计算等级

规则如下：

| 分数范围 | 等级 |
| :---: | :---: |
| 分数 >= 90 | A |
| 分数 >= 75 | B |
| 分数 >= 60 | C |
| 分数 < 60 | D |

代码实现：

```python
def calc_grade(score):
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"

print(calc_grade(100))
print(calc_grade(50))
print(calc_grade(70))
```

输出结果：

```python
A
D
C
```

### 1.4.5. 练习2：判断回文串

回文串指正着读和反着读都一样的字符串。

例如：

```python
level
radar
黄山落叶松叶落山黄
```

代码实现：

```python
def is_palindrome(s):
    return s == s[::-1]

print(is_palindrome("level"))
print(is_palindrome("hello"))
```

输出结果：

```python
True
False
```

### 1.4.6. 练习3：秒数转换为时分秒

需求：定义一个函数，将传入的秒数转换为小时、分钟、秒。

```python
def time_change(total_seconds):
    """
    该函数用于时间转换，将秒转换为小时、分钟、秒
    :param total_seconds: 总秒数
    :return: 小时，分钟，秒
    """
    hours = total_seconds // 3600
    minutes = (total_seconds - hours * 3600) // 60
    seconds = total_seconds - hours * 3600 - minutes * 60
    return hours, minutes, seconds

print(time_change(10000))

h, m, s = time_change(10000)
print(f"10000秒转换为{h}小时{m}分{s}秒")
```

输出结果：

```python
(2, 46, 40)
10000秒转换为2小时46分40秒
```

### 1.4.7. 练习4：判断三角形类型

需求：根据传入的三角形三条边，判断三角形类型。

类型包括：

1. 等边三角形。
2. 等腰三角形。
3. 普通三角形。
4. 不能构成三角形。

代码实现：

```python
def triangle_type(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a:
        return "不能构成三角形"
    elif a == b == c:
        return "等边三角形"
    elif a == b or a == c or b == c:
        return "等腰三角形"
    else:
        return "普通三角形"

print(triangle_type(3, 3, 3))
print(triangle_type(3, 3, 5))
print(triangle_type(3, 4, 5))
print(triangle_type(1, 2, 3))
```

# 2. 函数进阶

## 2.1. 函数变量的作用域

### 2.1.1. 什么是变量作用域

变量作用域指的是变量的作用范围，也就是变量在哪里可以使用，在哪里不可以使用。

变量可以分为：

| 类型 | 定义位置 | 作用范围 |
| :---: | :--- | :--- |
| 全局变量 | 函数外部 | 整个文件中都可以使用 |
| 局部变量 | 函数内部 | 只能在函数内部使用 |

### 2.1.2. 全局变量与局部变量

```python
num = 100

def circle_area(r):
    pi = 3.14
    area = pi * r * r
    return area

count = 0

c_area = circle_area(10)
print(c_area)
```

说明：

- `num` 和 `count` 是全局变量。
- `pi` 和 `area` 是局部变量。
- 局部变量只能在函数内部使用。
- 函数执行完毕后，局部变量会被销毁。

### 2.1.3. global 关键字

`global` 关键字用于在函数内部声明使用全局变量。

不使用 `global` 的情况：

```python
num1 = 1

def fun1():
    num1 = 100
    print(num1)

fun1()
print(num1)
```

输出结果：

```python
100
1
```

说明：函数内部的 `num1` 是局部变量，不会影响外部的全局变量。

使用 `global` 的情况：

```python
num1 = 1

def fun1():
    global num1
    num1 = 100
    print(num1)

fun1()
print(num1)
```

输出结果：

```python
100
100
```

说明：使用 `global num1` 后，函数内部修改的是全局变量 `num1`。

### 2.1.4. global 使用注意事项

1. 使用 `global` 时，要先声明，再使用。
2. 尽量避免在函数中大量修改全局变量。
3. 优先通过函数参数和返回值传递数据。
4. `global` 常用于程序状态、配置和计数器等场景。

## 2.2. 函数参数详解

### 2.2.1. 传参方式

传参方式指的是：调用函数时，传递实参的方式。

常见传参方式包括：

1. 位置参数。
2. 关键字参数。
3. 默认参数。
4. 不定长参数。

### 2.2.2. 位置参数

位置参数指的是：调用函数时，按照函数定义时参数的位置顺序传递数据。

```python
def reg_stu(name, age, gender, city):
    print(f"注册成功,姓名:{name}, 年龄:{age}, 性别:{gender}, 城市:{city}")
    return {"name": name, "age": age, "gender": gender, "city": city}

stu = reg_stu("张三", 18, "男", "北京")
print(stu)
```

输出结果：

```python
注册成功,姓名:张三, 年龄:18, 性别:男, 城市:北京
{'name': '张三', 'age': 18, 'gender': '男', 'city': '北京'}
```

位置参数的特点：

1. 写法简洁。
2. 参数顺序必须完全一致。
3. 参数较多时可读性较差，容易传错。

### 2.2.3. 关键字参数

关键字参数指的是：调用函数时，通过 `形参名=值` 的方式传递参数。

```python
def reg_stu(name, age, gender, city):
    print(f"注册成功,姓名:{name}, 年龄:{age}, 性别:{gender}, 城市:{city}")
    return {"name": name, "age": age, "gender": gender, "city": city}

stu = reg_stu(name="张三", age=18, gender="男", city="北京")
print(stu)

stu2 = reg_stu(gender="男", name="王五", city="上海", age=22)
print(stu2)
```

关键字参数的特点：

1. 参数顺序没有要求。
2. 可读性更强。
3. 参数较多时更容易维护。

### 2.2.4. 位置参数与关键字参数混用

位置参数和关键字参数可以混用，但必须满足：

> 位置参数在前，关键字参数在后。

正确写法：

```python
stu = reg_stu("赵四", 28, gender="男", city="上海")
print(stu)
```

错误写法：

```python
# 错误：位置参数不能放在关键字参数后面
stu = reg_stu(name="赵四", 28, gender="男", city="上海")
```

### 2.2.5. 两种传参方式的使用建议

| 传参方式 | 优点 | 缺点 | 适用场景 |
| :---: | :--- | :--- | :--- |
| 位置参数 | 简洁 | 可读性较差，容易出错 | 参数较少，且顺序自然 |
| 关键字参数 | 可读性强，易维护 | 代码略繁琐 | 参数较多，或容易混淆的场景 |

建议：

> 如果半年后回头看今天写的代码，不能一眼看出每个参数的含义，就应该使用关键字参数。

### 2.2.6. 默认参数

默认参数也叫缺省参数，用于在定义函数时为参数提供默认值。

调用函数时：

1. 如果没有传入该参数，就使用默认值。
2. 如果传入了该参数，就使用传入的值。

示例：

```python
def reg_stu(name, age, gender, city="北京"):
    print(f"注册成功,姓名:{name}, 年龄:{age}, 性别:{gender}, 城市:{city}")
    return {"name": name, "age": age, "gender": gender, "city": city}

stu = reg_stu("张三", 18, "男")
print(stu)

stu = reg_stu("赵四", 22, "男", "深圳")
print(stu)
```

输出结果：

```python
注册成功,姓名:张三, 年龄:18, 性别:男, 城市:北京
{'name': '张三', 'age': 18, 'gender': '男', 'city': '北京'}
注册成功,姓名:赵四, 年龄:22, 性别:男, 城市:深圳
{'name': '赵四', 'age': 22, 'gender': '男', 'city': '深圳'}
```

默认参数注意事项：

1. 默认参数必须放在没有默认值的参数后面。
2. 一个函数可以设置多个默认参数。
3. 调用函数时，如果传递了默认参数对应的值，就会覆盖默认值。

错误写法：

```python
# 错误：默认参数不能放在普通参数前面
def reg_stu(city="北京", name, age):
    pass
```

正确写法：

```python
def reg_stu(name, age, city="北京"):
    pass
```

### 2.2.7. 不定长参数

不定长参数也叫可变参数，用于函数定义和调用时参数个数不确定的场景。

不定长参数分为两类：

| 类型 | 写法 | 接收数据类型 | 作用 |
| :---: | :---: | :---: | :--- |
| 不定长位置参数 | `*args` | 元组 | 接收多个位置参数 |
| 不定长关键字参数 | `**kwargs` | 字典 | 接收多个关键字参数 |

### 2.2.8. 不定长位置参数：*args

`*args` 会接收所有匹配的位置参数，并封装成一个元组。

```python
def calc_data(*args):
    min_data = min(args)
    max_data = max(args)
    avg_data = sum(args) / len(args)
    return min_data, max_data, round(avg_data, 1)

data = calc_data(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
print(data)

data = calc_data(100, 200, 300, 400, 500)
print(data)
```

输出结果：

```python
(10, 100, 55.0)
(100, 500, 300.0)
```

注意：

1. `args` 是元组类型。
2. `args` 只是约定俗成的变量名，不是关键字。
3. 可以写成 `*data`，但一般推荐使用 `*args`。

### 2.2.9. 不定长关键字参数：**kwargs

`**kwargs` 会接收所有关键字参数，并封装成一个字典。

```python
def calc_data(*args, **kwargs):
    min_data = min(args)
    max_data = max(args)
    avg_data = sum(args) / len(args)

    if kwargs.get("round"):
        avg_data = round(avg_data, kwargs.get("round"))

    return min_data, max_data, avg_data

data = calc_data(100, 200, 300, 400, round=2, count=0)
print(data)

data = calc_data(33, 11, 28, 91, 32, 75, 49)
print(data)
```

输出结果：

```python
(100, 400, 250.0)
(11, 91, 45.57142857142857)
```

注意：

1. `kwargs` 是字典类型。
2. `kwargs` 只是约定俗成的变量名，不是关键字。
3. 可以写成 `**options`，但一般推荐使用 `**kwargs`。

### 2.2.10. *args 与 **kwargs 的应用场景

| 参数类型 | 适用场景 |
| :---: | :--- |
| `*args` | 处理数量不确定的数据 |
| `**kwargs` | 处理数量不确定的选项或配置参数 |

### 2.2.11. 函数的参数类型

函数的参数可以是普通数据，也可以是函数。

普通参数包括：

1. 数字。
2. 布尔。
3. 字符串。
4. 列表。
5. 元组。
6. 集合。
7. 字典。

示例1：数字作为参数。

```python
def circle_area(r):
    area = 3.14 * r ** 2
    return area

area = circle_area(10)
print(area)
```

示例2：列表作为参数。

```python
def calc_score(score_list):
    max_s = max(score_list)
    min_s = min(score_list)
    avg_s = round(sum(score_list) / len(score_list), 1)
    return max_s, min_s, avg_s

s_list = [589, 609, 605, 643, 677, 455, 477, 489, 503]
max_score, min_score, avg_score = calc_score(s_list)

print(max_score, min_score, avg_score)
```

### 2.2.12. 函数作为参数

函数本身也可以作为另一个函数的参数。

这种写法常用于把“要执行的逻辑”传递给函数。

```python
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def calc(x, y, oper):
    return oper(x, y)

result = calc(10, 20, add)
print(result)
```

输出结果：

```python
30
```

说明：

- `x` 和 `y` 传递的是实际要计算的数据。
- `oper` 传递的是函数中封装的计算逻辑。

## 2.3. 匿名函数

### 2.3.1. 什么是匿名函数

匿名函数指没有名称的函数，需要通过 `lambda` 表达式声明。

匿名函数适合逻辑简单、只在一个地方使用的场景。

### 2.3.2. 匿名函数语法

```python
lambda 参数列表: 函数体
```

注意：

1. 匿名函数通常只能写单行表达式。
2. 匿名函数可以有返回值，也可以没有返回值。
3. 返回结果时不需要写 `return`，表达式的结果就是返回值。

### 2.3.3. 命名函数写法

```python
def out_line():
    print("-------------------------")

def add(x, y):
    return x + y

out_line()
print(add(10, 20))
```

### 2.3.4. 匿名函数写法

```python
out_line = lambda: print("-------------------------")
add = lambda x, y: x + y

out_line()
print(add(100, 200))
```

输出结果：

```python
-------------------------
300
```

### 2.3.5. 命名函数与匿名函数的选择

| 函数类型 | 适用场景 |
| :---: | :--- |
| 命名函数 | 逻辑复杂、需要多步操作、需要重复使用、需要写说明文档 |
| 匿名函数 | 逻辑简单、只在一个地方使用、常作为高阶函数参数 |

建议：

> 代码的可读性和可维护性比简洁性更重要。

## 2.4. 案例实操

### 2.4.1. 案例1：N 的阶乘

需求：定义一个函数，根据传入的数字，计算该数字的阶乘。

阶乘示例：

```python
8! = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
7! = 7 * 6 * 5 * 4 * 3 * 2 * 1
6! = 6 * 5 * 4 * 3 * 2 * 1
```

递归公式：

```python
f(n) = n * f(n - 1)
f(1) = 1
```

代码实现：

```python
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(8))
```

输出结果：

```python
40320
```

### 2.4.2. 案例2：班级成绩统计

需求：根据输入的班级名称，以及班级中各个学员的考试总分，统计班级平均分、高于平均分的人数和低于平均分的人数。

```python
def calc_grade(class_name, *args):
    """
    根据输入的班级名称，以及班级中各个学员的考试总分，
    统计班级平均分、高于平均分的人数和低于平均分的人数
    :param class_name: 班级名称
    :param args: 班级学员分数，元组类型
    :return: 平均分，高于平均分的人数，低于平均分的人数
    """
    total_score = sum(args)
    avg_score = total_score / len(args)
    above_avg_count = sum(1 for score in args if score > avg_score)
    below_avg_count = sum(1 for score in args if score < avg_score)
    return avg_score, above_avg_count, below_avg_count

avg_score, above_avg_count, below_avg_count = calc_grade(
    "六年级一班",
    650, 683, 706, 593, 623, 634, 712, 588, 562, 470, 611
)

print(f"班级【六年级一班】的平均分是 {avg_score:.2f}, 高于平均分的人数有 {above_avg_count} 人, 低于平均分的人数有 {below_avg_count} 人")
```

### 2.4.3. 案例3：电商订单计算器

需求：定义一个函数，根据传入的一批商品信息、优惠信息和运费信息，计算订单总金额。

商品信息格式：

```python
("商品名", 价格, 数量)
```

规则：

1. 优惠券需要商品金额满 5000 才可以使用。
2. 优惠券金额不能超过商品总价。
3. 积分抵扣需要商品金额满 5000 才可以使用。
4. 100 积分抵扣 1 元。
5. 积分只能整百抵扣。
6. 抵扣金额不能超过商品总价。

基础代码示例：

```python
def calc_cart_total_price(*goods, coupon_price=0, point_deduction=0, express_price=0):
    """
    根据商品信息、优惠券、积分抵扣和运费计算购物车总金额
    :param goods: 商品信息
    :param coupon_price: 优惠券金额
    :param point_deduction: 积分抵扣金额
    :param express_price: 运费金额
    :return: 购物车总金额
    """
    total_price = sum(good[1] * good[2] for good in goods)

    if total_price > 5000:
        total_price -= coupon_price

    if total_price > 8000:
        total_price -= point_deduction

    return total_price + express_price

print(calc_cart_total_price(
    ("鼠标", 100, 2),
    ("键盘", 200, 3),
    coupon_price=500,
    point_deduction=1000,
    express_price=50
))
```

### 2.4.4. 列表推导式与生成器表达式

列表推导式的写法：

```python
[要插入的值 for i in 数据集 if 条件]
```

示例：

```python
nums = [i for i in range(10) if i % 2 == 0]
print(nums)
```

输出结果：

```python
[0, 2, 4, 6, 8]
```

特点：

1. 立即求值。
2. 一次性生成整个列表。
3. 会占用较多内存空间。

适用场景：

1. 数据量不大。
2. 需要立即生成全部数据。
3. 需要重复使用、多次访问。

生成器表达式的写法：

```python
(要插入的值 for i in 数据集 if 条件)
```

示例：

```python
nums = (i for i in range(10) if i % 2 == 0)
print(nums)

for num in nums:
    print(num)
```

特点：

1. 惰性求值。
2. 按需逐个生成元素。
3. 不会一次性把所有元素存入内存。
4. 更节省内存。

适用场景：

1. 处理大数据集。
2. 避免一次性加载所有数据到内存。
3. 作为 `sum()`、`max()`、`min()` 等函数的参数。

示例：

```python
total = sum(i for i in range(1000000))
print(total)
```

# 3. 类型注解

## 3.1. 基本介绍

### 3.1.1. 什么是类型注解

类型注解是 Python 中的一种语法特性，用于明确标识变量、函数参数和返回值的数据类型。

使用类型注解的好处：

1. 代码结构更清晰。
2. 代码逻辑更安全。
3. 更易维护。
4. 代码自动提示更准确。
5. 可以提前发现潜在问题。

注意：

> Python 是动态类型语言，类型注解只是提示，不是强制约束。

### 3.1.2. 变量类型注解

不使用类型注解：

```python
a = 695
score = 98.5
hobby = "Python"
flag = True
pic = None

names = ["A", "C", "E"]
phones = {"13309091111", "15209109121"}
options = {"count": 0, "total": 0}
goods = ("手机", 5999, 1)
```

使用类型注解：

```python
a: int = 695
score: float = 98.5
hobby: str = "Python"
flag: bool = True
pic: None = None

names: list[str] = ["A", "C", "E"]
phones: set[str] = {"13309091111", "15209109121"}
options: dict[str, int] = {"count": 0, "total": 0}
goods: tuple[str, int, int] = ("手机", 5999, 1)
```

### 3.1.3. 常见类型注解写法

| 数据类型 | 写法 |
| :---: | :--- |
| 整数 | `int` |
| 浮点数 | `float` |
| 布尔 | `bool` |
| 字符串 | `str` |
| 空值 | `None` |
| 列表 | `list[int]`、`list[str]` |
| 集合 | `set[str]` |
| 元组 | `tuple[str, int, int]` |
| 字典 | `dict[str, int]` |
| 多种类型 | `str | int` |

### 3.1.4. 类型推断

类型推断指 Python 解释器自动推断变量、表达式或函数返回值数据类型的能力。

```python
a = 695
score = 98.5
hobby = "Python"
flag = True
pic = None
```

在变量直接赋值、变量运算、容器推导等场景中，解释器可以自动推断类型。

### 3.1.5. 类型注解小结

类型注解的写法：

```python
变量名: 数据类型 = 值
```

例如：

```python
name: str = "Tom"
age: int = 18
score: float = 98.5
```

类型注解的核心作用不是限制代码运行，而是提升代码的可读性、可维护性和编辑器提示效果。

## 3.2. 函数类型注解

### 3.2.1. 函数参数和返回值注解

为函数添加类型注解，主要是给函数参数和返回值添加类型说明。

基本语法：

```python
def 函数名(参数名: 参数类型) -> 返回值类型:
    函数体
```

### 3.2.2. 示例1：计算平均分

```python
def calc(scores: list[int]) -> float:
    return sum(scores) / len(scores)
```

说明：

- `scores: list[int]` 表示参数 `scores` 是一个整数列表。
- `-> float` 表示函数返回值是浮点数。

### 3.2.3. 示例2：返回多个结果

```python
def calc_data(scores: list[int]) -> tuple[int, int, float]:
    max_v = max(scores)
    min_v = min(scores)
    avg_v = sum(scores) / len(scores)
    return max_v, min_v, avg_v
```

说明：

- 返回值类型是 `tuple[int, int, float]`。
- 表示函数返回一个元组，元组中依次是整数、整数、浮点数。

### 3.2.4. 带类型注解的电商订单计算器

```python
def calc_cart_total_price(
    *goods: tuple[str, float, int],
    coupon_price: int = 0,
    point_deduction: int = 0,
    express_price: float = 0
) -> float:
    """
    定义一个用于根据商品信息、优惠券、积分抵扣、运费信息计算购物车总金额的函数
    :param goods: 商品信息
    :param coupon_price: 优惠券金额
    :param point_deduction: 积分抵扣积分数
    :param express_price: 运费金额
    :return: 购物车总金额
    """
    total_price = sum(good[1] * good[2] for good in goods)

    if total_price > 5000 and coupon_price <= total_price:
        total_price -= coupon_price

    if total_price > 5000 and point_deduction // 100 <= total_price:
        total_price -= point_deduction // 100

    return total_price + express_price

print(calc_cart_total_price(
    ("鼠标", 88.5, 2),
    ("键盘", 168.9, 3),
    coupon_price=500,
    point_deduction=1000,
    express_price=8.5
))
```

### 3.2.5. 函数类型注解小结

函数中类型注解的语法：

```python
def calc_data(scores: list[int]) -> tuple[int, int, float]:
    max_v = max(scores)
    min_v = min(scores)
    avg_v = sum(scores) / len(scores)
    return max_v, min_v, avg_v
```

其中：

- `scores: list[int]` 表示参数类型。
- `-> tuple[int, int, float]` 表示返回值类型。

对于需要团队协作开发和长期维护的项目，推荐使用类型注解。

# 4. 本章总结

## 4.1. 函数基础总结

1. 函数是可重复使用的代码片段。
2. 使用 `def` 定义函数。
3. 函数必须先定义，再调用。
4. 函数可以有参数，也可以没有参数。
5. 函数可以有返回值，也可以没有返回值。
6. `return` 只负责返回结果，不负责输出。
7. 输出结果需要使用 `print()`。
8. 函数说明文档可以提升代码可读性和可维护性。
9. 函数嵌套调用遵循后进先出的调用顺序。

## 4.2. 函数进阶总结

1. 函数内部定义的变量是局部变量。
2. 函数外部定义的变量是全局变量。
3. `global` 可以在函数内部声明使用全局变量。
4. 位置参数按顺序传递。
5. 关键字参数按 `形参名=值` 传递。
6. 默认参数可以给参数设置默认值。
7. `*args` 用于接收多个位置参数。
8. `**kwargs` 用于接收多个关键字参数。
9. 函数也可以作为参数传递。
10. 匿名函数使用 `lambda` 定义。

## 4.3. 类型注解总结

1. 类型注解可以提高代码可读性和可维护性。
2. 变量可以添加类型注解。
3. 函数参数和返回值也可以添加类型注解。
4. 类型注解只是提示，不是强制约束。
5. 团队协作开发和长期维护项目中，推荐使用类型注解。