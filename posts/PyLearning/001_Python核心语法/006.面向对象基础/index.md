---
title: Python核心语法06-面向对象基础
author: Hygen
description: Python面向对象基础学习
published: 2026-05-24
---

# 1. 面向对象概述

## 1.1. 面向过程与面向对象

### 1.1.1. 面向过程编程

**核心思想**：把一个需求分解成一系列要执行的步骤，然后按照步骤依次执行这些任务（关注的是流程、步骤）。

**适用场景**：面向过程编程非常直接，适合简单、线性的任务。

例如，盖房子的面向过程思路：
1. 平整地基
2. 地基打桩
3. 地基浇筑
4. 主体施工
5. 砌墙施工
6. 外墙施工
7. 室内硬装
8. 室内软装

### 1.1.2. 面向对象编程

**核心思想**：把一个人/物的特征和功能打包到一起，关注的是"谁来帮我做这件事儿"。

**对象**：可以理解为现实中具体的人/物在程序中的数字化身（万物皆对象）。

每个对象都包含两部分：
- **属性**：对象的特征（如颜色、油耗、马力、工种、年龄）
- **方法**：对象的功能/行为（如推平、挖掘、打桩、砌墙）

例如，盖房子的面向对象思路：
- 挖掘机对象：负责平整地基
- 打桩机对象：负责地基打桩
- 混凝土车对象：负责地基浇筑
- 工人对象：负责砌墙、装修等工作

## 1.2. 类与对象

### 1.2.1. 什么是类

**类**：描述的是一组具有相同属性（特征）和方法（功能/行为）的模板。

### 1.2.2. 什么是对象

**对象**：对象是类的实例，是基于类创建出来的（实例对象）。

**类与对象的关系**：
- 类是模板，对象是根据模板创建出来的具体实例
- 一个类可以创建无数个对象
- 创建对象的过程，也称为对象的实例化

就像月饼模具和月饼的关系：
- 月饼模具是类，定义了月饼的形状、花纹等特征
- 每个月饼是对象，具有模具定义的特征，但可以有不同的颜色、口味等属性

# 2. 类与对象的定义

## 2.1. 基本类定义

### 2.1.1. 类定义语法

```python
# 定义类
class 类名:
    pass

# 创建对象
对象名 = 类名()
```

**命名规范**：类名遵循大驼峰命名法，每个单词的首字母都是大写，单词之间没有分隔符，例如：`UserInfo`、`UserAccount`、`Car`。

### 2.1.2. 动态添加属性

创建对象后，可以动态地为对象添加属性：

```python
# 定义类
class Car:
    pass

# 创建对象
c1 = Car()

# 动态添加属性
c1.brand = "BMW"
c1.name = "X5"
c1.price = 500000

# 查看对象的所有属性
print(c1.__dict__)  # 输出: {'brand': 'BMW', 'name': 'X5', 'price': 500000}
```

**说明**：`__dict__`是Python中用户自定义类实例的一个特殊属性，用于以字典形式存储对象的属性。

## 2.2. 初始化方法`__init__`

### 2.2.1. `__init__`方法介绍

`__init__`是初始化方法，对象创建后自动调用，主要用于设置对象的初始状态（设置对象属性）。

### 2.2.2. `__init__`方法语法

```python
# 定义类
class 类名:
    def __init__(self, 参数列表):
       self.属性名 = 参数值
       self.属性名 = 参数值

# 创建对象
对象名 = 类名(参数列表)
```

### 2.2.3. `self`参数介绍

`self`是类中定义的方法的第一个参数，表示当前创建的实例对象。

在方法内部，通过`self.属性名`可以访问和修改对象的属性。

### 2.2.4. 示例

```python
# 定义类
class Car:
    def __init__(self, c_brand, c_name, c_price):
        self.brand = c_brand
        self.name = c_name
        self.price = c_price

# 创建对象
c1 = Car("BMW", "X5", 500000)

# 查看对象的所有属性
print(c1.__dict__)  # 输出: {'brand': 'BMW', 'name': 'X5', 'price': 500000}

# 访问对象属性
print(c1.brand)  # 输出: BMW
print(c1.name)  # 输出: X5
print(c1.price)  # 输出: 500000
```

**推荐**：在定义类时，尽量使用`__init__`方法来初始化对象的属性，而不是动态添加属性。

## 2.3. 小结

1. 定义类时，类名遵循**大驼峰命名法**。
2. `__init__`是初始化方法，对象创建时自动调用，主要用于设置对象的初始状态。
3. `self`是类中定义的方法的第一个参数，表示当前创建的实例对象。

# 3. 实例方法

## 3.1. 实例方法定义

定义在类中的函数称之为方法，实例方法是属于每个实例对象的方法。

### 3.1.1. 实例方法语法

```python
# 定义类
class 类名:
    def __init__(self, 形参列表):
       self.属性名 = 参数值
       self.属性名 = 参数值
    
    def 方法名(self, 形参列表):
       方法体
```

### 3.1.2. 实例方法调用

```python
对象名.方法名(实参)
```

**注意**：调用实例方法时，不需要传递`self`参数，Python会自动将当前对象传递给`self`。

## 3.2. 示例

```python
class Car:
    def __init__(self, brand, name, price):
        self.brand = brand
        self.name = name
        self.price = price
    
    def running(self):
        """汽车行驶方法"""
        print(f"{self.brand} {self.name} 正在高速行驶...")
    
    def total_cost(self, discount, rate):
        """计算提车总价"""
        return self.price * discount + self.price * rate

# 创建对象
c1 = Car("BMW", "X5", 500000)

# 调用实例方法
c1.running()  # 输出: BMW X5 正在高速行驶...

total_price = c1.total_cost(0.9, 0.1)
print(f"提车总价为: {total_price:.0f}")  # 输出: 提车总价为: 500000
```

# 4. 魔法方法

## 4.1. 魔法方法介绍

**魔法方法**是指Python中提供的以双下划线开头和结尾的特殊方法，用于定义类的特殊行为。

**特点**：魔法方法不需要我们手动调用，Python会在合适的时机自动调用。

## 4.2. 常用魔法方法

| 魔法方法 | 描述 |
| :---: | :--- |
| `__init__` | 初始化方法，对象创建时自动调用 |
| `__str__` | 字符串表示方法，使用`print()`打印对象时自动调用 |
| `__eq__` | 相等比较方法，使用`==`比较两个对象时自动调用 |
| `__lt__` | 小于比较方法，使用`<`比较两个对象时自动调用 |
| `__le__` | 小于等于比较方法，使用`<=`比较两个对象时自动调用 |
| `__gt__` | 大于比较方法，使用`>`比较两个对象时自动调用 |
| `__ge__` | 大于等于比较方法，使用`>=`比较两个对象时自动调用 |

## 4.3. 魔法方法示例

### 4.3.1. `__str__`方法

默认情况下，打印对象会输出对象的内存地址。通过重写`__str__`方法，可以自定义对象的字符串表示。

```python
class Car:
    def __init__(self, brand, name, price):
        self.brand = brand
        self.name = name
        self.price = price
    
    def __str__(self):
        return f"{self.brand} {self.name} 价格: {self.price}元"

c1 = Car("BMW", "X5", 500000)
print(c1)  # 输出: BMW X5 价格: 500000元
```

### 4.3.2. `__eq__`方法

默认情况下，使用`==`比较两个对象会比较它们的内存地址。通过重写`__eq__`方法，可以自定义对象的相等比较逻辑。

```python
class Car:
    def __init__(self, brand, name, price):
        self.brand = brand
        self.name = name
        self.price = price
    
    def __eq__(self, other):
        # 当品牌、名称和价格都相同时，认为两个对象相等
        return (self.price == other.price 
                and self.brand == other.brand 
                and self.name == other.name)

c1 = Car("BMW", "X5", 500000)
c2 = Car("BMW", "X5", 500000)
c3 = Car("Audi", "Q7", 600000)

print(c1 == c2)  # 输出: True
print(c1 == c3)  # 输出: False
```

### 4.3.3. `__lt__`方法

默认情况下，自定义对象之间不能进行大小比较。通过重写`__lt__`方法，可以自定义对象的小于比较逻辑。

```python
class Car:
    def __init__(self, brand, name, price):
        self.brand = brand
        self.name = name
        self.price = price
    
    def __lt__(self, other):
        # 按价格比较大小
        return self.price < other.price

c1 = Car("BMW", "X5", 500000)
c2 = Car("Audi", "Q7", 600000)

print(c1 < c2)  # 输出: True
print(c1 > c2)  # 输出: False
```

## 4.4. 完整示例

```python
class Car:
    def __init__(self, brand, name, price):
        self.brand = brand
        self.name = name
        self.price = price
    
    def running(self):
        print(f"{self.brand} {self.name} 正在高速行驶...")
    
    def __str__(self):
        return f"{self.brand} {self.name} {self.price}"
    
    def __eq__(self, other):
        return self.price == other.price and self.brand == other.brand and self.name == other.name
    
    def __lt__(self, other):
        return self.price < other.price

c1 = Car("BMW", "X5", 500000)
print(c1)  # 输出: BMW X5 500000

c2 = Car("BMW", "X5", 500000)
print(c2)  # 输出: BMW X5 500000

print(c1 == c2)  # 输出: True
print(c1 < c2)  # 输出: False
```

## 4.5. 小结

1. 魔法方法是Python中提供的`__xxx__`形式的特殊方法。
2. 魔法方法无需手动调用，Python会在合适的时机自动调用。
3. 常用的魔法方法有：
   - `__init__`：初始化对象
   - `__str__`：自定义对象的字符串表示
   - `__eq__`：自定义对象的相等比较逻辑
   - `__lt__`、`__le__`、`__gt__`、`__ge__`：自定义对象的大小比较逻辑

# 5. 实例属性与类属性

## 5.1. 属性分类

属性分为两类：
- **实例属性**：属于每个具体对象的属性，每个对象都是独立的（各个对象特有的数据）。
- **类属性**：属于类本身的属性，所有实例共享（所有对象共享的数据或配置）。

## 5.2. 实例属性

实例属性在`__init__`方法中定义，通过`self.属性名`访问。

每个对象的实例属性都是独立的，修改一个对象的实例属性不会影响其他对象。

## 5.3. 类属性

类属性在类的内部、方法的外部定义，通过`类名.属性名`访问。

所有对象共享类属性，修改类属性会影响所有对象。

## 5.4. 示例

```python
class Car:
    # 类属性
    wheel = 4  # 轮胎数量
    tax_rate = 0.1  # 购置税税率
    
    def __init__(self, c_brand, c_name, c_price):
        # 实例属性
        self.brand = c_brand
        self.name = c_name
        self.price = c_price
    
    def running(self):
        print(f"{self.brand} {self.name} 正在高速行驶...")
    
    def total_cost(self):
        """计算提车总价"""
        return self.price + self.price * Car.tax_rate

# 创建两个对象
c1 = Car("BYD", "汉", 180000)
c2 = Car("Tesla", "Model Y", 260000)

# 访问实例属性
print(c1.brand)  # 输出: BYD
print(c2.brand)  # 输出: Tesla

# 访问类属性
print(Car.wheel)  # 输出: 4
print(c1.wheel)  # 输出: 4
print(c2.wheel)  # 输出: 4

# 修改类属性
Car.tax_rate = 0.08

# 所有对象都会受到影响
print(c1.total_cost())  # 输出: 194400.0
print(c2.total_cost())  # 输出: 280800.0
```

**说明**：通过实例查找属性时，会先查找实例属性，实例属性不存在时，再查找类属性。

# 6. 案例实操：教务管理系统

## 6.1. 需求分析

采用面向对象的编程思想，完成教务管理系统的开发。教务管理系统可以管理在校学生的成绩信息，通过控制台菜单与用户交互，具体功能如下：
1. 添加学生成绩：根据输入的学生姓名、语文成绩、数学成绩、英语成绩，记录在系统中
2. 修改学生成绩：根据输入的学生姓名，修改对应的学生成绩
3. 删除学生成绩：根据输入的学生姓名，删除对应的学生成绩
4. 查询指定学生成绩：根据输入的学生姓名，查找对应的学生成绩，并输出
5. 展示全部学生成绩：展示出系统中所有学生的成绩

## 6.2. 类设计

- `Student`类：表示学生，包含学生的姓名、语文成绩、数学成绩、英语成绩等属性，以及计算总分、平均分等方法
- `EduManagement`类：表示教务管理系统，包含学生列表属性，以及添加、修改、删除、查询、展示学生成绩等方法

## 6.3. 代码实现

```python
class Student:
    """学生类"""
    def __init__(self, name, chinese, math, english):
        self.name = name
        self.chinese = chinese
        self.math = math
        self.english = english
    
    def total_score(self):
        """计算总分"""
        return self.chinese + self.math + self.english
    
    def avg_score(self):
        """计算平均分"""
        return self.total_score() / 3
    
    def __str__(self):
        return (f"姓名: {self.name}, 语文: {self.chinese}, 数学: {self.math}, "
                f"英语: {self.english}, 总分: {self.total_score()}, 平均分: {self.avg_score():.1f}")

class EduManagement:
    """教务管理系统类"""
    def __init__(self):
        self.students = {}  # 学生字典，key是学生姓名，value是Student对象
    
    def add_student(self, student):
        """添加学生"""
        if student.name in self.students:
            print(f"学生{student.name}已存在！")
            return False
        self.students[student.name] = student
        print(f"学生{student.name}添加成功！")
        return True
    
    def update_student(self, name, chinese, math, english):
        """修改学生成绩"""
        if name not in self.students:
            print(f"学生{name}不存在！")
            return False
        student = self.students[name]
        student.chinese = chinese
        student.math = math
        student.english = english
        print(f"学生{name}成绩修改成功！")
        return True
    
    def delete_student(self, name):
        """删除学生"""
        if name not in self.students:
            print(f"学生{name}不存在！")
            return False
        del self.students[name]
        print(f"学生{name}删除成功！")
        return True
    
    def query_student(self, name):
        """查询学生成绩"""
        if name not in self.students:
            print(f"学生{name}不存在！")
            return None
        return self.students[name]
    
    def show_all_students(self):
        """展示所有学生成绩"""
        if not self.students:
            print("系统中没有学生信息！")
            return
        print("=" * 70)
        print("所有学生成绩信息：")
        print("-" * 70)
        for student in self.students.values():
            print(student)
        print("=" * 70)

def main():
    """主函数"""
    edu_system = EduManagement()
    
    while True:
        print("\n" + "=" * 30)
        print("欢迎使用教务管理系统")
        print("1. 添加学生成绩")
        print("2. 修改学生成绩")
        print("3. 删除学生成绩")
        print("4. 查询指定学生成绩")
        print("5. 展示全部学生成绩")
        print("6. 退出系统")
        print("=" * 30)
        
        choice = input("请选择要执行的操作(1-6): ")
        
        if choice == "1":
            # 添加学生成绩
            name = input("请输入学生姓名: ")
            try:
                chinese = int(input("请输入语文成绩: "))
                math = int(input("请输入数学成绩: "))
                english = int(input("请输入英语成绩: "))
            except ValueError:
                print("成绩必须是数字！")
                continue
            
            student = Student(name, chinese, math, english)
            edu_system.add_student(student)
        
        elif choice == "2":
            # 修改学生成绩
            name = input("请输入要修改的学生姓名: ")
            try:
                chinese = int(input("请输入新的语文成绩: "))
                math = int(input("请输入新的数学成绩: "))
                english = int(input("请输入新的英语成绩: "))
            except ValueError:
                print("成绩必须是数字！")
                continue
            
            edu_system.update_student(name, chinese, math, english)
        
        elif choice == "3":
            # 删除学生成绩
            name = input("请输入要删除的学生姓名: ")
            edu_system.delete_student(name)
        
        elif choice == "4":
            # 查询指定学生成绩
            name = input("请输入要查询的学生姓名: ")
            student = edu_system.query_student(name)
            if student:
                print("-" * 50)
                print(student)
                print("-" * 50)
        
        elif choice == "5":
            # 展示全部学生成绩
            edu_system.show_all_students()
        
        elif choice == "6":
            # 退出系统
            print("感谢使用教务管理系统，再见！")
            break
        
        else:
            print("输入错误，请重新选择！")

if __name__ == "__main__":
    main()
```

# 7. 异常处理

## 7.1. 什么是异常

**异常**（也称为Bug）就是程序运行过程中出现的错误，它会中断程序的正常执行流程。

**常见异常类型**：
- `NameError`：变量未定义
- `TypeError`：类型错误
- `IndexError`：索引超出范围
- `KeyError`：字典中不存在指定的key
- `ValueError`：值错误

**异常的作用**：
- 保证数据、逻辑的正确性，避免程序执行混乱
- 在开发阶段，尽量发现更多的问题，尽早解决问题，保障程序正常执行

**注意**：异常不是坏东西，而是编写健壮程序的重要工具。

## 7.2. 异常处理

程序运行过程中出现异常，有两种处理方案：
1. 不做处理：整个程序因为一个Bug，中断执行
2. 捕获异常：按照我们自己的处理方式，处理完异常，程序继续执行

### 7.2.1. 异常处理语法

```python
try:
    可能出现异常的业务代码1
    可能出现异常的业务代码2
    ...
except [异常类型 as 变量名]:
    出现异常时的预案
[finally:
    不管是否出现异常，都会执行的代码]
```

### 7.2.2. 基本示例

```python
try:
    print("=" * 30)
    print(my_name)  # my_name未定义，会抛出NameError
    print("=" * 30)
except NameError as e:
    print("程序运行报错，错误信息: ", e)
finally:
    print("释放资源 ~")
```

输出结果：
```
==============================
程序运行报错，错误信息:  name 'my_name' is not defined
释放资源 ~
```

### 7.2.3. 捕获多个异常

```python
try:
    num = int(input("请输入一个数字: "))
    result = 10 / num
    print(f"10 / {num} = {result}")
except ValueError as e:
    print("输入错误，必须输入数字！")
except ZeroDivisionError as e:
    print("除数不能为0！")
except Exception as e:
    print("未知错误，具体信息: ", e)
finally:
    print("程序执行完毕")
```

## 7.3. 异常的传递

**异常传递**就是异常在函数调用中层层上报的过程，直到有人处理它，或者程序崩溃。

```python
def fun1():
    print("fun1 ... running ...")
    fun2()

def fun2():
    print("fun2 ... running ...")
    fun3()

def fun3():
    print("fun3 ... running ...")
    print(my_color)  # my_color未定义，会抛出NameError

if __name__ == '__main__':
    try:
        fun1()
    except NameError as e:
        print("捕获到异常: ", e)
```

输出结果：
```
fun1 ... running ...
fun2 ... running ...
fun3 ... running ...
捕获到异常:  name 'my_color' is not defined
```

**说明**：异常从`fun3`抛出，传递到`fun2`，再传递到`fun1`，最后在主函数中被捕获处理。

# 8. 本章总结

## 8.1. 面向对象核心概念

1. **对象**：现实中具体的人/物在程序中的数字化身，包含属性和方法。
2. **类**：描述一组具有相同属性和方法的对象的模板。
3. **实例化**：根据类创建对象的过程。

## 8.2. 类与对象

1. 使用`class`关键字定义类，类名遵循大驼峰命名法。
2. `__init__`方法是初始化方法，对象创建时自动调用，用于设置对象的初始属性。
3. `self`参数表示当前实例对象，在方法内部通过`self`访问对象的属性和方法。

## 8.3. 实例方法

1. 定义在类中的函数称为方法，实例方法属于每个实例对象。
2. 调用实例方法时，不需要传递`self`参数，Python会自动传递。

## 8.4. 魔法方法

1. 魔法方法是Python中提供的`__xxx__`形式的特殊方法。
2. 魔法方法无需手动调用，Python会在合适的时机自动调用。
3. 常用魔法方法：`__init__`、`__str__`、`__eq__`、`__lt__`等。

## 8.5. 属性

1. **实例属性**：属于每个具体对象的属性，每个对象独立拥有。
2. **类属性**：属于类本身的属性，所有实例共享。

## 8.6. 异常处理

1. 异常是程序运行过程中出现的错误，会中断程序的正常执行。
2. 使用`try-except-finally`语句捕获和处理异常。
3. 异常会在函数调用中层层传递，直到被处理或程序崩溃。