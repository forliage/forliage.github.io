---
title: "SQL"
lecture: 2
course: "db"
date: 2026-03-21
---

# Introduction

In the previous lecture, we learned about relational algebra—a concise and rigorous query language, but not so user-friendly for humans. Imagine if every database query required writing a long mathematical formula like $\Pi_{A} ( \sigma_{P} (r_1 \times r_2) )$, databases would probably never leave the labs of computer scientists.

The world needed a language closer to natural language, easier to understand and use. Therefore, in the 1970s, IBM's Don Chamberlin and Ray Boyce, based on Codd's relational model, designed SEQUEL (Structured English Query Language), which later evolved into today's **SQL (Structured Query Language)**.

**Core Design Philosophy of SQL:**
1. **Declarative:** Users only need to **declare** what result they want, without specifying the specific steps or algorithms to obtain the result. You tell the database the "destination," and the database's query optimizer is responsible for finding the "best path." This contrasts sharply with imperative languages ​​like C++ or Java.
2. **Closure:** SQL operations take tables as input and output as output. This allows queries to be nested and combined infinitely, building powerful data pipelines.
3. **Comprehensive:** SQL is not just a query language (DQL); it also integrates data definition (DDL), data manipulation (DML), and data control (DCL) functions, providing a one-stop database solution.

# I.DDL

Before querying data, we must first build a framework for it. DDL is the blueprint and tool we use to define the skeleton of the database.

## 1.1.`CREATE TABLE`

`CREATE TABLE` does more than just create a table; it defines all the attributes of an entity and the universal laws (constraints) it must abide by.

```sql
CREATE TABLE student (
    s_id     CHAR(9) PRIMARY KEY,
    s_name   VARCHAR(50) NOT NULL,
    dept_name VARCHAR(50),
    credits  NUMERIC(4, 1) CHECK (credits >= 0),
    FOREIGN KEY (dept_name) REFERENCES department(dept_name)
);
```

**Design Philosophy Analysis:**

* **Data Types:** This is the most basic constraint. Why differentiate between `CHAR(9)` and `VARCHAR(50)`?
   * `CHAR(n)`: Fixed length. Whether storing "CS" or "History", it occupies n characters of space. **Advantages:** Fast processing speed because the offset within the record is fixed. **Disadvantages:** Wastes space. Suitable for storing data of fixed length, such as student IDs and national ID numbers.
   * `VARCHAR(n)`: Variable length. Only occupies the space actually needed (plus a small amount of length information). **Advantages:** Saves space. **Disadvantages:** Slightly slower processing. Suitable for storing data of varying lengths, such as names and addresses.
   * `NUMERIC(p, d)` vs. `FLOAT`: This is a trade-off between **precision** and **performance**. `NUMERIC` is used for scenarios requiring precise calculations (such as finance), while `FLOAT` is an approximation, with fast calculation speed but a loss of precision. 
* **Integrity Constraints:** These are the "laws" of the data, enforced by the DBMS. They move data validation logic down from the application layer to the database layer, ensuring data consistency and correctness under all circumstances.
   * **NOT NULL:** Guarantees the existence of attribute values.
   * **PRIMARY KEY:** Implicitly includes `NOT NULL` and `UNIQUE`. It is the unique identifier of a tuple and the cornerstone of **entity integrity**.
   * **FOREIGN KEY:** Establishes reference relationships between tables and is the guardian of **referential integrity**. It ensures that relationships between tables are valid and not dangling.
   * **CHECK:** Provides custom business rule validation, ensuring **domain integrity**.

## **1.2 `ALTER TABLE` And `DROP TABLE`**

* `ALTER TABLE r ADD A D;`: Adds a new attribute to an existing relation.
* `ALTER TABLE r DROP A;`: Deletes an attribute. (Note: This is a costly and dangerous operation; some older systems do not even support it.)
* `DROP TABLE r;`: Completely deletes all information (data, structure, indexes) of a relation. This is an irreversible operation.

#### **1.3 `CREATE INDEX`**

**设计哲学：逻辑与物理的分离。** 索引对用户是透明的，它不改变表的逻辑结构，也不影响查询的**结果**，只影响查询的**性能**。它就像一本书的目录，允许DBMS在不扫描整张表的情况下快速定位数据。

```
CREATE UNIQUE INDEX student_name_idx ON student (s_name);
```

*   `UNIQUE`关键字确保了该索引对应的列（或列组合）的值是唯一的，这是一种强制唯一性的高效方式。

### **第二部分：数据查询语言 (DQL)**

这是SQL最核心、最迷人的部分。我们将解构`SELECT`语句，并理解其背后严谨的逻辑执行顺序。

#### **2.1 `SELECT`语句的解剖与逻辑执行流**

一个看似简单的SQL查询，背后有一个严格的逻辑处理流程。理解这个流程是写对、写好任何复杂查询的关键。

```
SELECT DISTINCT T.dept_name, AVG(T.salary) AS avg_salary
FROM instructor AS T
WHERE T.salary > 50000
GROUP BY T.dept_name
HAVING COUNT(*) > 5
ORDER BY avg_salary DESC;
```

**逻辑执行顺序 (Mental Model):**

1.  **`FROM`**: **数据源**。确定查询涉及哪些表。如果有多张表，此时会形成一个逻辑上的笛卡尔积。`AS`关键字在这里用于创建别名，是代码可读性和处理自连接的关键。
2.  **`WHERE`**: **行过滤器 (Row Filter)**。对`FROM`子句产生的每一行数据进行逐一判断，只保留满足条件的行。**注意：** 此阶段无法使用聚合函数（如`AVG`, `COUNT`），因为它操作的是单行数据，分组还未发生。
3.  **`GROUP BY`**: **分组**。将通过`WHERE`筛选的行，按照指定的列进行分组，形成多个“行的集合”。后续的操作都将针对这些“组”进行。
4.  **`HAVING`**: **组过滤器 (Group Filter)**。对`GROUP BY`产生的每个组进行判断，只保留满足条件的组。这里是聚合函数大显身手的地方。**`HAVING`与`WHERE`的本质区别**：`WHERE`过滤行，`HAVING`过滤组。
5.  **`SELECT`**: **投影与计算**。确定最终结果集要包含哪些列。可以包含分组列、聚合函数计算结果、或其它表达式。`AS`在这里用于命名输出列。
6.  **`DISTINCT`**: **去重**。在`SELECT`之后，对结果集进行去重处理。
7.  **`ORDER BY`**: **排序**。对最终结果集进行排序，这纯粹是为了展示。可以使用`ASC`（升序，默认）或`DESC`（降序）。

#### **2.2 基本子句详解**

*   **`SELECT`**:
    *   `*` : 选择所有列。方便，但在生产环境中应避免，因为它降低了代码可读性，且可能传输不必要的数据。
    *   `DISTINCT` vs. `ALL` (默认): `DISTINCT`强制执行集合语义（去重），而SQL默认是多重集语义（保留重复）。**设计哲学：** 默认保留重复是出于性能考虑，去重操作（通常需要排序或哈希）是昂贵的。
*   **`WHERE`**:
    *   丰富的谓词：`=, <, >, LIKE, BETWEEN, IN`等。
    *   `LIKE '%Ze%'`: `%`匹配任意字符串，`_`匹配任意单个字符。
*   **`ORDER BY`**: 可以对多个列进行排序，并为每个列指定不同的排序方向。

### **第三部分：连接与集合操作**

#### **3.1 连接关系 (`JOIN`)：水平方向的融合**

**设计哲学：从隐式到显式。** 早期的SQL连接是在`FROM`子句中列出多个表，在`WHERE`子句中写连接条件。

```
-- 旧式（隐式）连接
SELECT * FROM student, takes WHERE student.s_id = takes.s_id;
```

这种写法的**巨大缺陷**在于，如果你忘记了`WHERE`条件，查询不会报错，而是会执行一个庞大的笛卡尔积，返回无意义的结果，这被称为“意外的交叉连接”。

为了解决这个问题，ANSI SQL-92标准引入了**显式`JOIN`语法**，这是现代SQL的最佳实践。

```
-- 新式（显式）连接
SELECT * FROM student INNER JOIN takes ON student.s_id = takes.s_id;
```

**优点：**

*   **清晰**：连接条件和过滤条件在语法上被分开了（`ON` vs `WHERE`）。
*   **安全**：不可能意外地忘记连接条件。

**连接的类型：**

*   **`INNER JOIN`**: 只返回两个表中能匹配上的行。
*   **`OUTER JOIN`**:
    *   **`LEFT OUTER JOIN`**: 返回左表的所有行，以及右表中能匹配上的行。如果右表没有匹配行，则右表的列显示为`NULL`。**应用场景**：查询所有学生及其选课情况（包括没有选课的学生）。
    *   **`RIGHT OUTER JOIN`**: 与`LEFT JOIN`相反。
    *   **`FULL OUTER JOIN`**: 返回左表和右表的所有行，在无法匹配的地方用`NULL`填充。
*   **`NATURAL JOIN`**: 自动在两个表的同名列上进行等值连接。**强烈不推荐在生产代码中使用！** 因为它过于依赖列名，如果表结构发生变化（如新增了一个同名列），查询逻辑可能在不经意间被破坏。

#### **3.2 集合操作：垂直方向的融合**

这些操作要求参与运算的查询结果具有相同的列数和兼容的数据类型。

*   **`UNION` / `UNION ALL`**: 合并两个结果集。`UNION`会自动去重（集合并），`UNION ALL`保留所有行（多重集并），性能更高。
*   **`INTERSECT` / `INTERSECT ALL`**: 返回两个结果集的交集。
*   **`EXCEPT` / `EXCEPT ALL`**: 返回在第一个结果集中存在，但在第二个结果集中不存在的行（集合差）。

### **第四部分：嵌套与聚合**

#### **4.1 聚合函数 (`AVG`, `COUNT`, `SUM`, `MIN`, `MAX`)**

这些函数是数据分析的基石。它们将一个“组”的数据压缩成一个单一的值。

一个重要的细节：除了`COUNT(*)`，所有聚合函数都会**忽略`NULL`值**。`COUNT(*)`统计所有行，而`COUNT(column)`统计该列中非`NULL`值的数量。

#### **4.2 嵌套子查询 (Subqueries)**

一个查询可以嵌套在另一个查询的`WHERE`, `FROM`, `SELECT`子句中，这极大地增强了SQL的表达能力。

*   **`WHERE`子句中的子查询**:
    *   `IN / NOT IN`: 判断值是否存在于子查询返回的（单列）集合中。
    *   `EXISTS / NOT EXISTS`: 判断子查询是否返回**任何行**。通常用于**相关子查询 (Correlated Subquery)**，即子查询的执行依赖于外层查询的当前行。`EXISTS`通常比`IN`更高效，因为它找到一个匹配后就可以停止。
    *   比较运算符 (`=, <, >`) 与 `ANY` / `ALL` 结合：例如 `> ANY(...)` 表示大于子查询结果中的任意一个（即大于最小值），`> ALL(...)` 表示大于子查询结果中的所有值（即大于最大值）。

#### **4.3 派生关系：`WITH`子句 (Common Table Expressions - CTEs)**

**设计哲学：告别“意大利面条式”查询。** 当子查询嵌套层级过多时，代码会变得极难阅读和维护。`WITH`子句允许你为子查询命名，像定义临时变量一样，然后在主查询中引用它们。

```
WITH cs_students AS (
    SELECT s_id, s_name
    FROM student
    WHERE dept_name = 'Comp. Sci.'
),
student_courses_count AS (
    SELECT s_id, COUNT(*) AS num_courses
    FROM takes
    GROUP BY s_id
)
SELECT cs.s_name, scc.num_courses
FROM cs_students cs
JOIN student_courses_count scc ON cs.s_id = scc.s_id;
```

**优点：**

*   **可读性**：将复杂查询分解为逻辑上独立的步骤。
*   **可维护性**：易于修改和调试每个逻辑单元。
*   **递归能力**：`WITH RECURSIVE`可以用来处理树形或图形结构数据，这是普通子查询无法做到的。

### **第五部分：视图与修改**

#### **5.1 视图 (`VIEW`)：数据的窗户**

视图是一个**虚拟表**，其内容由查询定义。它本身不存储数据，而是像一个动态的窗口，通过它可以看到基表中的数据。

**设计哲学与用途：**

1.  **简化复杂性**：将一个复杂的多表连接查询封装成一个简单的视图，供用户使用。
2.  **安全性**：可以创建一个只暴露部分列（隐藏工资列）或部分行（只显示本部门员工）的视图，对不同用户授予不同视图的权限。
3.  **逻辑数据独立性**：如果基表的结构发生变化（如拆分了一张表），可以重建视图来保持与旧结构相同的接口，而使用该视图的应用程序代码无需修改。

**视图更新的挑战：** 对视图的`INSERT`, `UPDATE`, `DELETE`操作最终要映射到对基表的修改。如果视图定义复杂（如包含聚合、`GROUP BY`、`DISTINCT`等），DBMS将无法确定如何明确地修改基表，因此这类视图是**不可更新**的。

#### **5.2 数据修改语言 (DML)**

*   **`INSERT INTO ... VALUES ...`**: 插入单行或多行指定值。
*   **`INSERT INTO ... SELECT ...`**: 将一个查询的结果批量插入到表中。
*   **`UPDATE ... SET ... WHERE ...`**: 更新满足条件的行的列值。**切记：** 如果忘记`WHERE`子句，将更新**整张表**！
*   **`DELETE FROM ... WHERE ...`**: 删除满足条件的行。同样，忘记`WHERE`子句将删除**所有数据**！

这些操作都会受到表上定义的完整性约束的检查。

### **第六部分：`NULL`**

`NULL` 可能是SQL中最令人困惑、也最富争议的设计。它不等于0，不等于空字符串，它不等于任何值，甚至不等于它自己 (`NULL = NULL` 的结果不是`TRUE`)。

*   **含义**：未知 (Unknown)、不适用 (Not Applicable)、缺失 (Missing)。
*   **三值逻辑 (Three-Valued Logic)**: SQL的逻辑运算有三种结果：`TRUE`, `FALSE`, `UNKNOWN`。
    *   `'A' = 'A'` -> `TRUE`
    *   `'A' = 'B'` -> `FALSE`
    *   `'A' = NULL` -> `UNKNOWN`
*   **`WHERE`子句的行为**: `WHERE`子句只保留谓词计算结果为`TRUE`的行，`FALSE`和`UNKNOWN`的行都会被丢弃。
*   **正确的`NULL`判断**: 必须使用 `IS NULL` 和 `IS NOT NULL`。