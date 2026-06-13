# Python 类及其最新用法详解

> 面向 **Python 3.12+ / 3.13** 的类（class）系统说明，并结合 CPython 官方文档整理当前推荐写法。  
> 参考：[类教程](https://docs.python.org/3/tutorial/classes.html) · [数据模型](https://docs.python.org/3/reference/datamodel.html) · [dataclasses](https://docs.python.org/3/library/dataclasses.html) · [typing](https://docs.python.org/3/library/typing.html) · [3.12 新特性 PEP 695](https://docs.python.org/3/whatsnew/3.12.html)

---

## 一、类是什么：对象模型速览

在 Python 里，**一切皆对象**；`class` 用来定义**用户自定义类型**的蓝图。

```python
class Dog:
    species = "Canis familiaris"   # 类属性（所有实例共享）

    def __init__(self, name: str) -> None:
        self.name = name             # 实例属性

    def bark(self) -> str:
        return f"{self.name}: 汪!"


d = Dog("小白")
print(d.bark())  # 小白: 汪!
```

| 概念 | 说明 |
|------|------|
| **实例** | `Dog("小白")` 创建的对象 |
| **`self`** | 实例方法的第一个参数，指向当前实例 |
| **`__init__`** | 初始化方法，在 `__new__` 创建实例之后调用 |
| **类属性** | 定义在类体上，通过 `Dog.species` 或 `d.species` 访问 |
| **实例属性** | 通常挂在 `self.xxx` 上，每个实例一份 |

**重要规则（派生类）：** 子类若自定义 `__init__`，应显式调用父类初始化：

```python
class WorkingDog(Dog):
    def __init__(self, name: str, job: str) -> None:
        super().__init__(name)   # 推荐写法
        self.job = job
```

`__init__` **不能**返回非 `None` 的值；返回值由 `__new__` 负责。

---

## 二、面向对象核心机制

### 1. 继承与 MRO（方法解析顺序）

```python
class Animal:
    def speak(self) -> str:
        return "..."


class Cat(Animal):
    def speak(self) -> str:
        return "喵"


class RobotCat(Cat):
    def speak(self) -> str:
        return super().speak() + " (合成音)"
```

- 多继承时，MRO 由 **C3 线性化** 决定，可用 `RobotCat.__mro__` 查看。
- 改行为优先用 **`super()`**，避免硬编码父类名。

### 2. 封装：命名约定

| 写法 | 含义 |
|------|------|
| `name` | 公开 |
| `_name` | 约定「内部使用」，仍可直接访问 |
| `__name` | 名称改写（name mangling），减少子类意外覆盖 |

真正需要访问控制时，配合 **`@property`** 或 **descriptor**，而不是只靠下划线。

### 3. `@property`：把方法当属性用

```python
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if value < 0:
            raise ValueError("半径不能为负")
        self._radius = value

    @property
    def area(self) -> float:
        import math
        return math.pi * self._radius ** 2
```

底层依赖 **描述符协议**（`__get__` / `__set__` / `__delete__`）。内置 `property` 就是描述符的标准实现。

### 4. 常用特殊方法（dunder）

| 方法 | 典型用途 |
|------|----------|
| `__repr__` / `__str__` | 调试/打印表示 |
| `__eq__` / `__hash__` | 相等比较、能否放进 `set`/`dict` 键 |
| `__len__` / `__getitem__` | 像容器一样使用 |
| `__enter__` / `__exit__` | 上下文管理器 `with` |
| `__call__` | 让实例可被「调用」 |

示例：

```python
class Stack(list):
    def push(self, item):
        self.append(item)

    def __repr__(self) -> str:
        return f"Stack({list(self)!r})"
```

---

## 三、现代写法 1：`@dataclass`（数据类）

当类**主要是存数据**、需要 `__init__`、`__repr__`、`__eq__` 时，优先用标准库 `dataclasses`（3.7+，3.10+ 功能更全）。

### 基础

```python
from dataclasses import dataclass


@dataclass
class Employee:
    name: str
    dept: str
    salary: int


john = Employee("john", "computer lab", 1000)
print(john)  # Employee(name='john', dept='computer lab', salary=1000)
```

装饰器等价于显式打开 `init=True, repr=True, eq=True` 等默认选项。

### 默认值与「可变默认值陷阱」

**错误**（所有实例共享同一个 list）：

```python
# 不要这样写
@dataclass
class Bad:
    items: list = []
```

**正确**：用 `field(default_factory=...)`

```python
from dataclasses import dataclass, field


@dataclass
class Good:
    items: list = field(default_factory=list)


assert Good().items is not Good().items  # 每个实例独立列表
```

字段顺序规则：**无默认值的字段**必须排在**有默认值**的字段前面。

### 常用参数（3.10+ 很实用）

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class Point:
    x: float
    y: float
    z: float = 0.0


p = Point(x=1.5, y=2.5)  # 关键字-only 构造（kw_only=True）
# p.x = 3  # frozen=True 时会报错
```

| 参数 | 作用 |
|------|------|
| `frozen=True` | 实例不可变（类似 namedtuple） |
| `slots=True` | 生成 `__slots__`，省内存、属性访问更快 |
| `kw_only=True` | 生成仅关键字参数的 `__init__` |
| `order=True` | 自动生成 `<`, `<=` 等比较方法 |
| `repr=False` 等 | 用 `field(repr=False)` 细粒度控制 |

---

## 四、现代写法 2：类型系统与「结构化类」

Python 的类不只用于 OOP，也用于**静态类型标注**（mypy、Pyright）。

### 1. PEP 695：类级泛型（3.12+，推荐新语法）

旧写法要 `TypeVar` + `Generic`；新写法直接在类名后写类型参数：

```python
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()


ints = Stack[int]()
ints.push(1)
```

多参数、约束、上界也可写在方括号里：

```python
from collections.abc import Sequence


class WeirdTrio[T, B: Sequence[bytes], S: (int, str)]:
    ...
```

### 2. `Protocol`：结构化子类型（鸭子类型的类型版）

不要求继承，只要**有对应方法**就符合协议：

```python
from typing import Protocol


class Drawable(Protocol):
    def draw(self) -> None: ...


class Circle:
    def draw(self) -> None:
        print("画圆")


def render(shape: Drawable) -> None:
    shape.draw()


render(Circle())  # OK，无需继承 Drawable
```

泛型 Protocol（3.12+）：

```python
class GenProto[T](Protocol):
    def meth(self) -> T: ...
```

适合：**插件接口、测试 mock、解耦具体类**。

### 3. `TypedDict`：带类型的字典结构

运行时仍是普通 `dict`，类型检查器负责校验键和值类型：

```python
from typing import TypedDict


class Point2D(TypedDict):
    x: int
    y: int
    label: str


a: Point2D = {"x": 1, "y": 2, "label": "good"}
```

适合：**JSON/API 载荷、配置对象**，不想引入完整类层次时。

### 4. `@dataclass_transform`（3.11+）

给**自己的装饰器/元类**打上标记，让类型检查器知道它像 `@dataclass` 一样生成 `__init__` 等：

```python
from typing import dataclass_transform


@dataclass_transform()
def create_model[T](cls: type[T]) -> type[T]:
    # 运行时做字段处理...
    return cls


@create_model
class CustomerModel:
    id: int
    name: str
```

常见于 **Pydantic、attrs 风格框架** 与静态分析的配合。

---

## 五、现代写法 3：性能与内存

### `__slots__` / `@dataclass(slots=True)`

限制实例只能有固定属性，减少 `__dict__` 开销：

```python
@dataclass(slots=True)
class LightUser:
    id: int
    name: str
```

大量实例（百万级）时效果明显；但不能再随意 `obj.new_attr = 1`。

### 不可变数据

- `@dataclass(frozen=True)`：逻辑不可变
- 需要真正常量语义时，还可配合 `enum`、或只暴露只读接口

---

## 六、其他常见类模式（简表）

| 模式 | 场景 | 示例 |
|------|------|------|
| **`enum.Enum` / `StrEnum`** | 固定常量集合 | `class Status(StrEnum): OK = "ok"` |
| **`abc.ABC` + `@abstractmethod`** | 强制子类实现接口 | 模板方法、插件基类 |
| **元类 `type`** | 创建类时改结构 | ORM、DSL（高级，慎用） |
| **命名元组 `NamedTuple`** | 轻量不可变记录 | 比 dataclass 更轻 |
| **Pydantic `BaseModel`** | 校验 + 序列化 API 模型 | Web/API（第三方，生态主流） |

---

## 七、描述符与「方法绑定」（理解即可）

- **实例方法**：`obj.method` 会通过描述符绑定 `self`，本质是 `MethodType(func, obj)`。
- **`@classmethod` / `@staticmethod`**：分别绑定类、不绑定实例。
- 自定义 `Property` 类可实现与内置 `property` 相同协议（`__get__`/`__set__`/`__delete__`）。

日常开发**很少手写描述符**；知道 `property` 和 `dataclass.field` 背后机制即可。

---

## 八、怎么选：2025–2026 实践建议

```
需要定义类型?
    │
    ├─ 主要是数据字段? ──是──► @dataclass (+ slots/frozen)
    │       │
    │       └─ 要 JSON/API 校验? ──是──► Pydantic BaseModel
    │
    ├─ 仅结构、无行为 ──► TypedDict
    │
    └─ 需要多态/接口? ──是──► 普通 class + ABC 或 Protocol
            │
            └─ 否则 ──► 普通 class / 泛型 class Foo[T]
```

| 需求 | 推荐 |
|------|------|
| DTO、配置、领域实体（纯数据） | `@dataclass(slots=True)` |
| 对外 API、严格校验 | Pydantic v2 `BaseModel` |
| 鸭子类型、可插拔 | `Protocol` |
| JSON 形状固定 | `TypedDict` |
| 容器/算法类 | 普通 class + 泛型 `class Foo[T]` |
| 大量实例、省内存 | `slots=True` |
| 业务逻辑、复杂状态机 | 普通 class + 清晰继承/组合 |

**组合优于过深继承**：用 `Protocol` 定义能力，用组合持有依赖，比 5 层继承树更易维护。

---

## 九、完整现代示例（综合运用）

```python
from dataclasses import dataclass, field
from typing import Protocol
from collections.abc import Sequence


# 1. Protocol 定义能力
class Summable(Protocol):
    def total(self) -> float: ...


# 2. 泛型 + dataclass 存数据
@dataclass(frozen=True, slots=True, kw_only=True)
class LineItem:
    name: str
    price: float
    qty: int = 1

    @property
    def subtotal(self) -> float:
        return self.price * self.qty


@dataclass(slots=True)
class Order[T: LineItem]:
    items: Sequence[T] = field(default_factory=list)

    def total(self) -> float:
        return sum(i.subtotal for i in self.items)


# 3. 使用
def print_total(obj: Summable) -> None:
    print(obj.total())


order = Order(items=[
    LineItem(name="书", price=29.9, qty=2),
    LineItem(name="笔", price=5.0),
])
print_total(order)  # Order 满足 Summable（有 total 方法）
```

---

## 十、与版本相关的新特性小结

| 特性 | 版本 | 说明 |
|------|------|------|
| `@dataclass` | 3.7+ | 数据类标准方案 |
| `dataclass(slots=..., kw_only=...)` | 3.10+ | 更省内存、更安全构造 |
| `@dataclass_transform` | 3.11+ | 第三方 dataclass 式 API 的类型支持 |
| `class Foo[T]` 泛型语法 | **3.12+ (PEP 695)** | 替代大部分 `TypeVar` + `Generic` 写法 |
| `Protocol[T]` 简写 | 3.12+ | 泛型协议更简洁 |
| 持续改进的 `typing` | 3.13+ | 类型系统与标准库文档持续更新 |

---

## 延伸阅读方向

- **描述符与元类**（底层机制）
- **Pydantic v2 模型 vs dataclass**（Web 开发）
- **多继承与 MRO 实战**
- **用 Pyright 给现有类补全类型**

---

*文档生成日期：2026-05-26*
