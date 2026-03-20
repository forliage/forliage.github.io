---
title: "The Relational Model"
lecture: 1
course: "db"
date: 2026-03-20
---

# Introduction

Before the advent of the relational model, what was the database world like? It was dominated by **network models** and **hierarchical models**. Imagine data as a complex maze, and application programmers, like explorers, carefully navigate between data records, guided by "pointers" or "paths."

* **Hierarchical Model:** Data is organized into a tree structure. Its advantages include clear description of one-to-many relationships and high query efficiency. However, its disadvantage is fatal: if two entities have a many-to-many relationship (e.g., a student can choose multiple courses, and a course can be chosen by multiple students), it must be simulated using redundant data, which is extremely cumbersome.
* **Network Model:** Allows multiple parent nodes, solving the many-to-many problem. However, it introduces complex "linked lists of pointers" to maintain relationships between records. What problems does this cause?
* **Navigational Access:** Programmers must explicitly specify in the code how to "navigate" from one record to the next. Query logic is tightly coupled to the physical storage path of the data.
* **Poor Data Independence:** Once the underlying data storage structure changes (e.g., adding a new index or path for optimization), all application code accessing this data may need to be rewritten. This is a maintenance nightmare.

**A Design Philosophy Question:** Can we create a model where users only care about "**What data do they want (What)**", without worrying about "**How ​​do they get the data (How)**?"

This is the core problem that Edgar F. Codd addressed in his groundbreaking 1970 paper, "A Relational Model of Data for Large Shared Data Banks." His answer was the **relational model**. Its core idea is to organize data using a simple, intuitive, and mathematically sound structure, leaving the complex relationships between data to the database management system (DBMS), thus freeing programmers from the maze.

# I. Relation

## 1.1.Core Structure: From Mathematics to Tables

* **Domain:** A collection of atomic values. For example, the set of all integers, or the set of all strings less than 20 characters long.
* **Design Philosophy: The Importance of Atomicity (First Normal Form, 1NF)**
* **Why Atomicity?** If a field can be further divided (e.g., an "Address" field containing "Country-Province-City"), then you cannot independently query, sort, or index "City." Atomicity ensures that the value of each cell in a data table is the smallest indivisible unit of information, which greatly simplifies the logic of data processing. This is the first important design decision made by the relational model for simplicity.
* **Further Thoughts:** Modern databases (such as PostgreSQL) support JSON or array types. Does this violate atomicity? Yes, to some extent. This is a trade-off made to address the needs of semi-structured data storage, sacrificing some of the purity of the model for flexibility.
* **Relation:**
* **Formal Definition:** Given a set of domains $D_1, D_2, \ldots, D_n$, a **relation** $r$ is a **subset** of the Cartesian product $D_1 \times D_2 \times \ldots \times D_n$.
* **Intuitive Understanding:** A relation is a **two-dimensional table**.
* **Tuple:** A **row** in a relation (table). It represents a real-world entity or a relationship between entities.
* **Attribute:** A **column** in a relation (table). Each attribute has a name and a corresponding domain.

## 1.2.Schema v.s. Instance

* **Schema:** This is the "blueprint" or "skeleton" of a relation, defining its name, attributes, and domains. For example: `Student(Student ID: CHAR(10), Name: VARCHAR(20), Major: VARCHAR(30))`. We usually represent it as $R(A_1, A_2, \ldots, A_n)$.

* **Instance:** This is the **specific set** of tuples in a relation at a given moment, i.e., the actual data in the table. Instances change over time (insertions, deletions, and modifications), while the schema is usually stable.

**Design Philosophy: Separation of Blueprint and Architectural Structure** This separation is the core embodiment of data independence. Applications are written to a stable "schema," while the DBMS manages the constantly changing "architecture." Regardless of data additions or deletions, as long as the blueprint remains unchanged, the application does not need to be modified.

## 1.3.Intrinsic properties of relationships: disorder and uniqueness

Codd's relational model directly inherits from the mathematical theory of sets, giving it two crucial characteristics:

1. **Tuple Unorderedness:** As a set, tuples in a relation have no inherent order. The first row and the fifth row are essentially the same.
2. **Tuple Uniqueness:** A fundamental definition of set theory states that a set has no duplicate elements. Therefore, no two identical tuples are allowed in a relation.

**Design Philosophy:** Why Insist on "Sets"?

* **Logical Data Independence:** "Unorderedness" means you can store or retrieve tuples in any order without affecting the correctness of the query results. The DBMS is free to optimize access paths based on physical storage (such as indexes), without the user needing to worry.
* **Entity Integrity:** "Uniqueness" guarantees that each row in a table uniquely identifies a real-world entity. If duplicate rows exist, which Zhang San does the record "Student Zhang San" refer to? The data will become ambiguous. This characteristic is enforced through **keys**.

# II.Integrity Constraints

If the relational model merely provides a table structure, it's not powerful enough. Its true power lies in its ability to define and enforce the "laws" of the data world—integrity constraints.

## 2.1 Keys: Unique Identifiers of Entities

How to ensure the uniqueness of tuples? Through "keys".

* **Superkey:** A set of one or more attributes whose values ​​can **uniquely** identify a tuple. For example, in the `students` table, `{student ID}` is a superkey, and `{student ID, name}` is also a superkey.
* **Candidate Key:** The **minimum** superkey. That is, removing any attribute from the set of attributes will cause it to cease to be a superkey. For example, `{student ID}` is a candidate key, while `{student ID, name}` is not, because removing `name` still makes `{student ID}` a superkey. If `{ID number}` can also uniquely identify a student, then it is also a candidate key.
* **Primary Key:** The primary key selected by the database designer from one or more candidate keys, used as the primary identifier for tuples in the relation.
* **Design Philosophy:** Why Specify a Primary Key?
    1. **Explicitness:** Provides a clear, unambiguous, "official" ID for each row in the table.
    2. **Performance:** DBMSs typically create indexes automatically for the primary key to greatly accelerate lookup and join operations based on it.
    3. **Reference:** The primary key is the "anchor" (via foreign keys) for other tables to reference records in this table.
* **Best Practice:** Generally, choose a "man-made" key with no business meaning (such as an auto-incrementing ID or UUID) as the primary key, rather than a key with business meaning (such as an ID number), because business requirements may change (e.g., an ID number may be upgraded).

## 2.2 Foreign Key: The Bridge Between Relationships

If the database has only one table, it will be meaningless. Foreign keys are the mechanism for establishing relationships between tables and are central to maintaining **referential integrity**.

* **Definition:** A set of attributes $FK$ in relation $r_1$ that references the primary key $PK$ of relation $r_2$. This means that every value of $FK$ in $r_1$ must either be equal to the $PK$ value of some tuple in $r_2$ or must be NULL.

* $r_1$ is called a **Referencing Relation**.
* $r_2$ is called a **Referenced Relation**.

**Design Philosophy: Eliminating Dangling Pointers** The foreign key mechanism fundamentally solves the "dangling pointer" problem in network/hierarchical models. It guarantees:

* You cannot insert a record into a `course selection table` whose `student ID` is in a non-existent `student table`.

You cannot delete a student from the `Students` table if that student's `Student ID` is still referenced in the `Course Selection` table (unless a cascading delete strategy is defined).

This ensures data consistency and validity, freeing the responsibility of data validation from thousands of application code snippets and placing it uniformly under the DBMS's guarantee.

# III.Relational Algebra

Relational algebra is a language for operating on relations, forming the theoretical foundation of query languages ​​such as SQL. Each operation takes one or more relations as input and outputs a **new relation**. This "closure" property (the result of an operation is still a similar object) allows operations to be nested and combined arbitrarily, constructing complex queries.

## 3.1 Six Fundamental Operations

These six operations constitute the complete set of relational algebra operations, theoretically capable of expressing all relational queries.

1. **Selection - $\sigma$**

   * **Purpose:** Filters tuples (rows).
   * **Syntax:** $\sigma_p(r)$, where $p$ is the selection predicate (condition).
   * **Relationship with SQL:** The `WHERE` clause.
   * **Example:** Query tuples where $A='\beta'$ and $D>5$. $\sigma_{A='\beta' \land D>5}(r)$ 

$$ 
r=\begin{array}{|c|c|c|c|}
 \hline
  A & B & C & D \\ 
  \hline 
  \alpha & \alpha & 1 & 7 \\ 
  \hline 
  \alpha & \beta & 5 & 7 \\ 
  \hline 
  \beta & \beta & 12 & 3\\ 
  \hline 
  \beta & \beta & 23 & 10 \\ 
  \hline 
\end{array} 
$$ 

$$ 
\sigma_{A=\beta \land D_{>5}}(r)=
\begin{array}{|c|c|c|c|} 
\hline A & B & C & D\\
\hline 
\beta &\beta & 23 & 10\\ 
\hline 
\end{array} 
$$

2. **Projection (Project) - $\Pi$**
   * **Function:** Selects attributes (columns).
   * **Syntax:** $\\Pi\_{A\_1, A\_2, \\ldots, A\_k}(r)$
   * **Relationship with SQL:** A list of column names after `SELECT`.
   * **Design Philosophy:** The result is still a set. **After the projection operation, duplicate tuples will be automatically **eliminated** because the output must be a valid relation (set). This is a key difference between relational algebra and typical SQL `SELECT` (SQL does not eliminate duplicates by default; `DISTINCT` is required).
   * **Example:** Project columns A and C from relation r. $\Pi_{A, C}(r)$ 
   $$ 
   r=\begin{array}{|c|c|c|} 
   \hline 
   A & B & C\\ 
   \hline 
   \alpha & 10 & 1\\ 
   \hline \alpha & 20 & 1 \\ 
   \hline 
   \beta & 30 & 1\\ 
   \hline \beta & 40 & 2\\ 
   \hline 
   \end{array} 
   $$ 
   $$ 
   \Pi_{A,C}(r)=
   \begin{array}{|c|c|} 
   \hline 
   A & C \\ 
   \hline 
   \alpha & 1\\ 
   \hline 
   \alpha & 1\\ 
   \hline 
   \beta & 1\\ 
   \hline \beta & 2\\ 
   \hline 
   \end{array} \Longrightarrow 
   \begin{array}{|c|c|} 
   \hline A & C \\ 
   \hline \alpha & 1\\ 
   \hline \beta & 1\\ 
   \hline \beta & 2\\ 
   \hline 
   \end{array} 
   $$

3. **Union - $\cup$**
   * **Purpose:** Merges tuples from two relations.
   * **Syntax:** $r \cup s$
   * **Requirements:** $r$ and $s$ must be **Union-compatible**, i.e.:
      1. They have the same number of attributes (same number of tuples).
      2. The corresponding attributes have the same (or compatible) domains.
   * **Relationship with SQL:** `UNION`.
   * **Example:** 
   $$ 
   r = \begin{array}{|c|c|} 
   \hline A & B \\ 
   \hline 
   \alpha & 1 \\ 
   \hline 
   \alpha & 2\\ 
   \hline 
   \beta & 1 \\ 
   \hline 
   \end{array}\quad s=\begin{array}{|c|c|} 
   \hline 
   A & B \\ \hline \alpha & 2 \\ \hline \beta & 3\\ \hline \end{array} 
   $$ 

   $$ 
   r \cup s=\begin{array}{|c|c|} 
   \hline 
   A & B \\ \hline 
   \alpha & 1 \\ \hline \alpha & 2 \\ \hline \beta & 1 \\ \hline \beta & 3 \\ \hline \end{array} 
   $$

4. **Set Difference - $-$**

   * **Purpose:** Subtracts a tuple from one relation that exists in another.
   * **Syntax:** $r - s$
   * **Requirements:** Must be compatible with both.
   * **SQL Connection:** `EXCEPT` (or `MINUS` in some SQL dialects).
   * **Example:** 
   $$ 
   r = \begin{array}{|c|c|} 
   \hline 
   A & B \\ \hline \alpha & 1 \\ \hline \alpha & 2\\ \hline \beta & 1 \\ \hline 
   \end{array}
   \quad s=
   \begin{array}{|c|c|} \hline A & B \\ \hline \alpha & 2 \\ \hline \beta & 3\\ \hline 
   \end{array} 
   $$ 
   $$ 
   r-s=\begin{array}{|c|c|} 
   \hline 
   A & B \\ \hline \alpha & 1 \\ \hline \beta & 1 \\ \hline 
   \end{array} 
   $$

5. **Cartesian Product - $\times$** 
   * **Function:** This generates a wider relation by performing all possible pairings on all tuples of two relations.
   * **Syntax:** $r \times s$
   * **Design Philosophy:** The foundation of all joins. The Cartesian product itself is rarely used directly because it produces a large number of meaningless combinations. However, it is the theoretical basis for all **join** operations. A join operation can be viewed as a **Cartesian product** followed by a **selection** operation.
   * **Example:** 
   $$ 
   r = \begin{array}{|c|c|} \hline A & B \\ \hline \alpha & 1 \\ \hline \beta & 2\\ \hline \end{array}\quad s=\begin{array}{|c|c|c|} \hline C & D & E\\ \hline \alpha & 10 & a \\ \hline \beta & 10 & a \\ \hline \beta & 20 & b \\ \hline \gamma & 10 & b \\ \hline \end{array} 
   $$ 

   $$ 
   r \times s = \begin{array}{|c|c|c|c|c|} 
   \hline A & B & C & D & E\\ 
   \hline \alpha & 1 & \alpha & 10 & a \\ 
   \hline \alpha & 1 & \beta & 10 & a \\ 
   \hline \alpha & 1 & \beta & 20 & b \\ 
   \hline \alpha & 1 & \gamma & 10 & b \\ 
   \hline \beta & 2 & \alpha & 10 & a \\ 
   \hline \beta & 2 & \beta & 10 & a \\ 
   \hline \beta & 2 & \beta & 20 & b \\ 
   \hline \beta & 2 & \gamma & 10 & b \\ 
   \hline 
   \end{array} 
   $$

   * **Note:** If $r$ and $s$ have attributes with the same name, you need to resolve the conflict by renaming them first.

6. **Rename (Rename) - $\rho$**
   * **Purpose:** Assigns a new name to a relation or its attributes.
   * **Syntax:** $\rho_x(E)$ renames the result of expression E to x; $\rho_{x(A_1, \ldots, A_n)}(E)$ renames the attribute while renaming it.
   * **Design Philosophy:** Resolves ambiguity and improves readability. This is a useful operation in relational algebra. Renaming is essential when performing self-joins (joining a table with itself) or dealing with Cartesian products of attributes with the same name.

## 3.2 Additional Operations

Although these operations can be expressed using combinations of basic operations, they are defined as independent operators due to their extremely high frequency of use. This not only simplifies things for users, but more importantly, it provides the query optimizer with crucial **intent information**.

1. **Intersection - $\cap$**
   * **Function:** Extracts the common tuples of two relations.
   * **Syntax:** $r \cap s$
   * **Equivalent Expression:** $r - (r - s)$ or $s - (s - r)$.
   * **Relationship with SQL:** `INTERSECT`.

2. **Join**
   * **Design Philosophy:** Why is a dedicated Join necessary? **While `Join` can be simulated using `Cartesian product + selection`, `Join` explicitly tells the DBMS: "I want a match based on specific conditions." When a DBMS sees a `Join` statement, it immediately uses an efficient join algorithm (such as Hash Join or Merge Join) instead of actually calculating a large Cartesian product. This is a crucial step from "mathematical theory" to "engineering implementation."
   * **$theta$-Join:** $r \bowtie_\theta s \equiv \sigma_\theta(r \times s)$
   * The most general join; $theta$ can be any comparison condition (such as `>`, `<=`).
   * **Natural Join:** $r \bowtie s$
   * This is a very important special case. Its execution steps are:
      1. Find all attributes with **the same name** in $r$ and $s$.
      2. Calculate $r \times s$.
      3. Select the tuples that have **equal values** on all attributes with the same name.
      4. Project the results, **removing duplicate attribute columns with the same name**.
   * **Advantages:** Concise.
   * **Disadvantages and Risks:** It's an "implicit" join, where the join condition is determined by the column names. If the table structure changes unexpectedly (e.g., someone adds columns with the same name to two unrelated tables), the query results may be incorrect without being noticed. Therefore, in modern SQL, it's recommended to explicitly specify the join condition using the `ON` clause.

3. **Division - $\div$**
   * **Purpose:** Handles queries of the "for all" type.
   * **Scenario:** "Query the student IDs of students who have taken **all** courses offered by the Computer Science Department."
   * **Intuitive Understanding:** $r(R) \div s(S)$, where the attribute set $S \subset R$. The resulting attributes are $R-S$. A tuple $t$ appears in the result if and only if for every tuple $u$ in $s$, the concatenated tuple $tu$ exists in $r$.
   * **Equivalent expression:** $r \div s = \Pi_{R-S}(r) - \Pi_{R-S}((\Pi_{R-S}(r) \times s) - r)$
   * This complex expression precisely illustrates the necessity of defining a separate operator for division.

4. **Assignment - $\leftarrow$**
   * **Purpose:** Saves the result of a query to a temporary relational variable for use in subsequent queries.
   * **Relationship with SQL:** With the `WITH` clause (Common Table Expressions, CTE) or creating a temporary view.
   * **Design Philosophy:** It implements a **procedural** approach to querying, enabling an extremely complex query to be broken down into a series of logically clear steps, greatly enhancing readability and maintainability.

# IV.Aggregation and Computation

Basic relational algebra can only filter and combine data. However, in practical applications, we also need to perform statistics and calculations.

## 4.1 Generalized Projection

Allows the use of arithmetic expressions in the projection list $\Pi$.

* **Syntax:** $\Pi_{F_1, F_2, \ldots, F_n}(E)$
* **Example:** Calculate the annual salary for each employee. Assume the `employee` table has a `monthly_salary` column.
   * $\Pi_{\text{name}, \text{monthly\_salary} * 12 \rightarrow \text{annual\_salary}}(\text{employee})$
* **Relationship with SQL:** Calculations can be performed directly in the `SELECT` list, such as `SELECT name, monthly_salary * 12 AS annual_salary FROM employee`. 

## 4.2 Aggregation Functions and Grouping
* **Aggregation Functions:** Take a set of values ​​as input and return a single value. Common ones include:
* `AVG` (average), `MIN` (minimum), `MAX` (maximum), `SUM` (summation), `COUNT` (counting).
* **Grouping Operator ($\mathcal{G}$):**
   * **Syntax:** $_{G_1, G_2, \ldots, G_n} \mathcal{G}_{F_1(A_1), F_2(A_2), \ldots, F_m(A_m)}(E)$
      * $G_1, \ldots, G_n$ are grouping properties.
      * $F_i(A_i)$ is the aggregation function and its properties.
   * **Relationship with SQL:**
      * The list of grouping attributes corresponds to the `GROUP BY` clause.
      * The list of aggregate functions corresponds to the aggregate expressions in the `SELECT` clause.
   * **Example:** Query the number of students and average age for each major.
      * $_{\text{major}} \mathcal{G}_{\text{COUNT(student ID)}, \text{AVG(age)}}(\text{student})$

# V.Database Modifications

Relational algebra can not only be used for querying, but also for precisely describing database modification operations. All modifications can be viewed as reassignments to relational variables.

1. **Deletion:**
   * **Logical:** Subtracts a set of tuples that satisfy a specific condition from the relation.
   * **Algebraic Representation:** $r \leftarrow r - E$, where $E$ is a relational algebra expression whose result is the set of tuples to be deleted.
   * **Example:** Delete all students majoring in Computer Science.
   * `students` $\leftarrow$ `students` - $\sigma\_{\text{major}='Computer Science'}(\text{students})$
   * **Relationship with SQL:** `DELETE FROM students WHERE major = 'Computer Science'`
2. **Insertion:**
   * **Logical:** Merges a new set of tuples into the relation.
   * **Algebraic Representation:** $r \leftarrow r \cup E$, where $E$ is the set of tuples to be inserted.
   * **Example:** Insert a new student.
   * `student` $\leftarrow$ `student` $\cup {('2025001', 'Chen Ming', 'Physics')}$
   * **Relationship with SQL:** `INSERT INTO ...`
3. **Update:**
   * **Logic:** This is the most complex. An update can be viewed as "conditionally replacing an old value with a new one." In relational algebra, this is usually modeled using **generalized projection**.
   * **Algebraic Representation:** $r \leftarrow \Pi\_{F_1, F_2, \ldots, F_n}(r)$
   * For attributes $A_i$ that do not need to be updated, $F_i$ is $A_i$ itself.
   * For the attribute $A_j$ that needs updating, $F_j$ is an expression that calculates the new value. This expression can use conditional logic of the `CASE` class.
   * **Example:** Change the major of all computer science students to "Artificial Intelligence".
   * This is a more complex expression, which can be imagined as:
      1. Select the tuples that need updating: $T_1 = \sigma_{\text{major}='Computer Science'}(\text{student})$
      2. Select the tuples that do not need updating: $T_2 = \sigma_{\text{major}<>'Computer Science'}(\text{student})$
      3. Calculate the new value for the tuples that need updating: $T'_1 = \Pi_{\text{student ID}, \text{name}, 'Artificial Intelligence' \rightarrow \text{major}}(T_1)$
      4. Merge the results: `Student` $\leftarrow T_2 \cup T'_1$
   * **Relationship with SQL:** `UPDATE Student SET Major = 'Artificial Intelligence' WHERE Major = 'Computer Science'`. The SQL syntax is obviously much simpler here, but the underlying logic is the same: locate the tuple, calculate the new value, and replace the old tuple.