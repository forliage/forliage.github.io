---
title: "论文解析:(OpenAI,2509)《Why Language Models Hallucinate》"
description: ""
pubDate: "2025-09-06"
heroImage: ""
---

# 论文解析:(OpenAI,2509)《Why Language Models Hallucinate》

[阅读原文，参考链接](https://cdn.openai.com/pdf/d04913be-3f6f-4d2b-b283-ff432ef4aaa5/why-language-models-hallucinate.pdf)

分为两大核心部分：

*   **第一部分：幻觉的统计力学基础**。此部分将建立论文的核心数学框架，形式化地证明幻觉现象在标准预训练范式下的统计必然性。我们将逐一展开并证明所有关键定理。
*   **第二部分：幻觉的社会动力学分析**。此部分将分析现行评估体系如何从博弈论角度激励并强化了幻觉行为，并对论文提出的改革方案进行形式化阐述。

### 第一部分：幻觉的统计力基础

本部分旨在从第一性原理出发，推导出幻觉现象的产生是统计学习过程的内在属性。

#### **1.1. 形式化框架：定义与公理**

我们首先建立分析所需的基础数学结构。

*   **定义 1.1.1 (样本空间 $\\mathcal{X}$)**: 令 $\\mathcal{X}$ 为一个离散的、有限的字符串集合，该集合包含了所有在句法上有效且在语义上貌似合理的字符串。此定义旨在将分析从语法层面提升至事实层面。
    
*   **公理 1.1.2 (真伪二分公理)**: 集合 $\\mathcal{X}$ 存在一个唯一的划分（Partition），将其分为两个不相交的子集：
    
    *   **有效集 $\\mathcal{V}$ (Valid Set)**: $\\mathcal{V} \\subset \\mathcal{X}$，其中所有字符串均为事实准确的。
    *   **错误集 $\\mathcal{E}$ (Error Set)**: $\\mathcal{E} \\subset \\mathcal{X}$，其中所有字符串均为事实错误的。 此划分满足 $\\mathcal{X} = \\mathcal{V} \\cup \\mathcal{E}$ 且 $\\mathcal{V} \\cap \\mathcal{E} = \\emptyset$。
*   **定义 1.1.3 (真实分布 $p$)**: 令 $p: \\mathcal{X} \\to \[0, 1\]$ 为一个理想的概率分布，称为真实分布。其支撑集 $\\text{supp}(p)$ 完全包含于 $\\mathcal{V}$ 中。形式化地，$\\forall x \\in \\mathcal{E}, p(x) = 0$。因此，$\\sum\_{x \\in \\mathcal{V}} p(x) = 1$。
    
*   **定义 1.1.4 (语言模型 $\\hat{p}$)**: 令 $\\hat{p}: \\mathcal{X} \\to \[0, 1\]$ 为一个通过在真实世界语料上训练得到的语言模型，它是一个旨在近似 $p$ 的概率分布。与 $p$ 不同，$\\hat{p}$ 的支撑集可能包含 $\\mathcal{E}$。
    
*   **定义 1.1.5 (生成错误率 $\\text{err}$)**: 语言模型 $\\hat{p}$ 的生成错误率 $\\text{err}$ 定义为 $\\hat{p}$ 分配给错误集 $\\mathcal{E}$ 的总概率质量： $$ \\text{err} := \\hat{p}(\\mathcal{E}) = \\sum\_{x \\in \\mathcal{E}} \\hat{p}(x) $$ 这是我们最终希望分析和建立下界的核心量。
    

#### **1.2. 核心分析工具：Is-It-Valid (IIV) 二元分类问题**

为了分析 $\\text{err}$，我们构造一个辅助的、更易于处理的监督学习问题。

*   **定义 1.2.1 (IIV 任务)**: IIV 任务是一个二元分类问题，其目标是学习一个函数 $f: \\mathcal{X} \\to {+, -}$，其中： $$ f(x) = \\begin{cases} + & \\text{if } x \\in \\mathcal{V} \\ - & \\text{if } x \\in \\mathcal{E} \\end{cases} $$
    
*   **定义 1.2.2 (IIV 测试分布 $\\mathcal{D}$)**: 为了理论分析，我们定义一个在 $\\mathcal{X}$ 上的混合分布 $\\mathcal{D}$。从 $\\mathcal{D}$ 中采样一个样本 $x$ 的过程如下：首先进行一次伯努利试验 $b \\sim \\text{Bernoulli}(0.5)$。若 $b=1$，则从 $p$ 中抽取一个样本 $x$；若 $b=0$，则从 $\\mathcal{E}$ 上的均匀分布 $\\text{Uniform}(\\mathcal{E})$ 中抽取一个样本 $x$。 因此，$\\mathcal{D}$ 的概率质量函数为： $$ D(x) = \\frac{1}{2} p(x) \\cdot \\mathbb{I}(x \\in \\mathcal{V}) + \\frac{1}{2 |\\mathcal{E}|} \\cdot \\mathbb{I}(x \\in \\mathcal{E}) $$ 其中 $\\mathbb{I}(\\cdot)$ 是指示函数。
    
*   **定义 1.2.3 (基于 $\\hat{p}$ 的贝叶斯最优分类器 $\\hat{f}$)**: 任何生成模型 $\\hat{p}$ 都可以被用作一个IIV分类器。在给定模型 $\\hat{p}$ 的情况下，我们可以构造一个近似的贝叶斯最优分类器 $\\hat{f}$。该分类器通过比较后验概率 $\\Pr(f(x)=+ | x)$ 和 $\\Pr(f(x)=- | x)$ 来决策。在我们的IIV设定下，这等价于比较 $p(x)/2$ 和 $1/(2|\\mathcal{E}|)$。由于我们无法直接使用 $p(x)$，我们用其近似 $\\hat{p}(x)$ 来构造分类器。分类规则为： $$ \\hat{f}(x) = \\begin{cases} + & \\text{if } \\hat{p}(x) > 1/|\\mathcal{E}| \\ - & \\text{if } \\hat{p}(x) \\le 1/|\\mathcal{E}| \\end{cases} $$ 这个分类器的决策边界是 $\\hat{p}(x) = 1/|\\mathcal{E}|$。
    
*   **定义 1.2.4 (IIV 误分类率 $\\text{err}\_{\\text{iiv}}$)**: 分类器 $\\hat{f}$ 在分布 $\\mathcal{D}$ 上的总错误概率： $$ \\text{err}\_{\\text{iiv}} := \\Pr\_{x \\sim \\mathcal{D}}\[\\hat{f}(x) \\neq f(x)\] = \\sum\_{x \\in \\mathcal{X}} D(x) \\cdot \\mathbb{I}(\\hat{f}(x) \\neq f(x)) $$
    

#### **1.3. 核心定理：生成错误率的下界**

现在我们陈述并证明连接 $\\text{err}$ 和 $\\text{err}\_{\\text{iiv}}$ 的核心定理。为保持通用性，我们直接进入包含提示（prompt）的场景。

*   **扩展定义 (带提示场景)**:
    
    *   样本 $x$ 现在是一个二元组 $(c, r)$，其中 $c \\in \\mathcal{C}$ 是提示， $r \\in \\mathcal{R}$ 是回复。
    *   真实分布 $p$ 现在是条件分布 $p(r|c)$，语言模型是 $\\hat{p}(r|c)$。
    *   $\\mathcal{V}\_c, \\mathcal{E}\_c$ 分别是给定提示 $c$ 时的有效和错误回复集。
    *   $\\text{err} := \\sum\_c \\mu(c) \\sum\_{r \\in \\mathcal{E}\_c} \\hat{p}(r|c)$，其中 $\\mu(c)$ 是提示的分布。
*   **定理 1.3.1 (Theorem 1)**: 对于任何真实分布 $p$ 和语言模型 $\\hat{p}$，以下不等式成立： $$ \\text{err} \\ge 2 \\cdot \\text{err}\_{\\text{iiv}} \\frac{\\max\_c |\\mathcal{V}\_c|}{\\min\_c |\\mathcal{E}\_c|} - \\delta $$ 其中，$$\\delta := |\\sum\_c \\mu(c) (\\sum\_{r \\in A\_c} \\hat{p}(r|c) - \\sum\_{r \\in A\_c} p(r|c))|$$ 是校准误差 $$A\_c = {r \\mid \\hat{p}(r|c) > 1/K}$$ 且 $$K = \\min\_c |\\mathcal{E}\_c|$$
    

**证明**:

令 $$K := \\min\_c |\\mathcal{E}\_c|$$ $$k := \\max\_c |\\mathcal{V}\_c|$$分类阈值为 $$T=1/K$$ 对于每个提示 $c$，定义 $$A\_c := {r \\mid \\hat{p}(r|c) > 1/K}$$ 和 $$B\_c := {r \\mid \\hat{p}(r|c) \\le 1/K}$$

**生成错误率分解**: $$ \\text{err} = \\sum\_c \\mu(c) \\left( \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c) + \\sum\_{r \\in B\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c) \\right) $$

**IIV 误分类率分解**: $$ \\text{err}\_{\\text{iiv}} = \\sum\_c \\mu(c) \\left( \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\frac{1}{2|\\mathcal{E}\_c|} + \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} \\frac{p(r|c)}{2} \\right) $$

**目标**: 建立 $\\sum\_c \\mu(c) \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c)$ 和 $\\sum\_c \\mu(c) \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\frac{1}{2|\\mathcal{E}\_c|}$ 的关系。 对于任何 $c$ 和 $r \\in A\_c \\cap \\mathcal{E}\_c$： \* 我们有 $\\hat{p}(r|c) > 1/K$。 \* 由 $K$ 的定义，$K \\le |\\mathcal{E}\_c|$，因此 $1/K \\ge 1/|\\mathcal{E}\_c|$。 \* 所以，$$\\hat{p}(r|c) > 1/K \\ge 1/|\\mathcal{E}\_c| = 2 \\cdot \\frac{1}{2|\\mathcal{E}\_c|}$$ 将上式两边乘以 $\\mu(c)$ 并对所有 $c$ 和 $r \\in A\_c \\cap \\mathcal{E}\_c$ 求和，得到： $$ \\sum\_c \\mu(c) \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c) > 2 \\sum\_c \\mu(c) \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\frac{1}{2|\\mathcal{E}\_c|} \\quad \\cdots (1) $$

**目标**: 建立 $\\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c)$ 和 $\\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} \\frac{p(r|c)}{2}$ 的关系。 首先，令 $$\\text{err}\_{\\text{iiv}, B} = \\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} \\frac{p(r|c)}{2}$$ $$2 \\cdot \\text{err}\_{\\text{iiv}, B} = \\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} p(r|c)$$ 由于 $p(r|c)=0$ 对于 $r \\in \\mathcal{E}\_c$，我们有 $$\\sum\_{r \\in B\_c} p(r|c) = \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} p(r|c)$$ 因此，$$2 \\cdot \\text{err}\_{\\text{iiv}, B} = \\sum\_c \\mu(c) \\sum\_{r \\in B\_c} p(r|c)$$ 引入校准误差 $\\delta$。设 $$\\hat{P}(B) = \\sum\_c \\mu(c) \\sum\_{r \\in B\_c} \\hat{p}(r|c)$$ 和 $$P(B) = \\sum\_c \\mu(c) \\sum\_{r \\in B\_c} p(r|c)$$那么 $$\\delta \\ge P(B) - \\hat{P}(B)$$ 所以 $$P(B) \\le \\hat{P}(B) + \\delta$$ 代入得到：$$2 \\cdot \\text{err}\_{\\text{iiv}, B} \\le \\hat{P}(B) + \\delta = \\sum\_c \\mu(c) \\left( \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} \\hat{p}(r|c) + \\sum\_{r \\in B\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c) \\right) + \\delta$$ 移项：$$\\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c) \\ge 2 \\cdot \\text{err}\_{\\text{iiv}, B} - \\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} \\hat{p}(r|c) - \\delta$$ **对负项进行上界放缩**: 对于 $r \\in B\_c \\cap \\mathcal{V}\_c$，我们有 $\\hat{p}(r|c) \\le 1/K$。每个 $c$ 最多有 $k$ 个这样的有效回复。 $$\\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{V}\_c} \\hat{p}(r|c) \\le \\sum\_c \\mu(c) \\cdot k \\cdot (1/K) = k/K$$ 代入得到第二个关键不等式： $$ \\sum\_c \\mu(c) \\sum\_{r \\in B\_c \\cap \\mathcal{E}\_c} \\hat{p}(r|c) \\ge 2 \\cdot \\text{err}\_{\\text{iiv}, B} - \\frac{k}{K} - \\delta \\quad \\cdots (2) $$

将不等式 (1) 和 (2) 相加，并令 $\\text{err}\_{\\text{iiv}, A} = \\sum\_c \\mu(c) \\sum\_{r \\in A\_c \\cap \\mathcal{E}\_c} \\frac{1}{2|\\mathcal{E}\_c|}$。 $$ \\text{err} \\ge 2 \\cdot \\text{err}\_{\\text{iiv}, A} + 2 \\cdot \\text{err}\_{\\text{iiv}, B} - \\frac{k}{K} - \\delta $$ 由于 $\\text{err}\_{\\text{iiv}} = \\text{err}\_{\\text{iiv}, A} + \\text{err}\_{\\text{iiv}, B}$，我们得到： $$ \\text{err} \\ge 2 \\cdot \\text{err}\_{\\text{iiv}} - \\frac{k}{K} - \\delta $$ 将 $k, K$ 的定义代回，即完成证明。

#### **1.4. IIV 困难性的来源分析**

**1.4.1. 任意事实与统计复杂性 (Theorem 2)**

*   **场景**: 事实之间无模式可循，学习等同于记忆。
*   **核心结论**: $\\text{err} \\gtrsim \\text{sr}$ (独有样本率)。
*   **详细证明逻辑 (附录 B)**:
    1.  **引理 1 (古德-图灵估计的扩展)**: 定义 Missing Mass (MM) 为模型遇到训练中未回答过的问题时，给出非IDK回答的总概率。证明独有样本率 sr 是 MM 的一个高概率的良好估计，即 $|\\text{MM} - \\text{sr}| \\le \\epsilon$ with high prob。
    2.  **核心论证 (独立性)**: 对于训练集中未见的查询 $c \\in \\mathcal{U}$，算法输出的 $\\hat{p}(\\cdot|c)$ 与真实答案 $a\_c$ 在统计上是独立的。
    3.  **计算期望IIV错误率**:
        *   令 $\\gamma\_c$ 为在查询 $c$ 上的IIV错误率贡献。
        *   $E\[\\gamma\_c\] = E \\left\[ \\frac{1}{2} \\mathbb{I}(\\hat{f}(c,a\_c)=-) + \\frac{1}{2|\\mathcal{E}\_c|} \\sum\_{r \\in \\mathcal{E}\_c} \\mathbb{I}(\\hat{f}(c,r)=+) \\right\]$。
        *   由于 $a\_c$ 是从 $\\mathcal{R}\_c$ 中均匀选取的，且与 $\\hat{f}$ 独立，可以证明 $E\[\\gamma\_c\] = 1/2$。
    4.  **集中性论证**: 在未见查询集上的总IIV错误率 $\\text{err}\_{\\text{iiv, unseen}} = \\sum\_{c \\in \\mathcal{U}} \\mu'(c) \\gamma\_c$。这是一个独立随机变量的和，其期望为 $E\[\\text{err}\_{\\text{iiv, unseen}}\] = \\text{MM}/2$。根据霍夫丁不等式，$\\text{err}\_{\\text{iiv, unseen}}$ 以高概率集中在其期望附近。
    5.  **整合**: $\\text{err} \\ge 2 \\text{err}\_{\\text{iiv}} - \\dots \\approx 2 \\text{err}\_{\\text{iiv, unseen}} \\approx 2 (\\text{MM}/2) = \\text{MM} \\approx \\text{sr}$。

### 第二部分：幻觉的社会动力学分析

#### **2.1. 评估范式的激励结构分析**

*   **定义 2.1.1 (二元评分机制)**: 一个评分函数 $g\_c: \\mathcal{R}\_c \\to {0, 1}$，其中存在一个唯一的正确答案 $a\_c$ 使得 $g\_c(a\_c)=1$，对于所有其他回答 $r \\neq a\_c$（包括错误和IDK），$g\_c(r)=0$。
    
*   **博弈论模型 (Observation 1)**:
    
    *   **代理人**: 语言模型，其目标是最大化期望得分。
    *   **信念**: 模型对哪个答案是正确答案 $a\_c$ 持有一个主观概率分布 $p\_c(r) := \\Pr(\\text{r is the correct answer } a\_c)$。
    *   **策略与期望收益**:
        *   **策略 S1 (回答 $r'$，其中 $r'$ 不是IDK)**: $$ E\[\\text{Score}(r')\] = \\sum\_{r \\in \\mathcal{R}\_c} p\_c(r) \\cdot g\_r(r') = p\_c(r') \\cdot 1 + (1-p\_c(r')) \\cdot 0 = p\_c(r') $$
        *   **策略 S2 (回答 $r\_{IDK}$)**: $$ E\[\\text{Score}(r\_{IDK})\] = p\_c(r\_{IDK}) = 0 $$ (假设IDK永远不是正确答案)。
    *   **理性决策**: 只要存在任何一个非IDK回答 $r'$ 使得模型的主观信念 $p\_c(r') > 0$，那么选择 $r'$ 的期望收益就严格大于选择IDK。
    *   **结论**: 在二元评分机制下，任何一个非完全无知的理性代理人，其最优策略集合中都不包含弃权选项。

#### **2.2. 解决方案：引入显式置信度目标的形式化描述**

*   **定义 2.2.1 (带惩罚的评分机制)**: 令 $t \\in (0,1)$ 为一个置信度阈值。定义一个新的评分函数 $g'\_c$: $$ g'\_c(r) = \\begin{cases} 1 & \\text{if } r = a\_c \\ 0 & \\text{if } r = r\_{IDK} \\ -t/(1-t) & \\text{otherwise} \\end{cases} $$
    
*   **新规则下的博弈论分析**:
    
    *   **策略 S1 (回答 $r' \\neq r\_{IDK}$)**: $\\begin{aligned} E\[\\text{Score}(r')\] &= p\_c(r') \\cdot g'\_{r'}(r') + (1-p\_c(r')) \\cdot g'\_{\\neg r'}(r') \\ &= p\_c(r') \\cdot 1 + (1-p\_c(r')) \\cdot \\left(-\\frac{t}{1-t}\\right) \\end{aligned}$
    *   **策略 S2 (回答 $r\_{IDK}$)**: $E\[\\text{Score}(r\_{IDK})\] = 0$。
*   **新的理性决策边界**: 代理人选择回答 $r'$ 当且仅当 $E\[\\text{Score}(r')\] > 0$。 $$ p\_c(r') \\cdot 1 + (1-p\_c(r')) \\cdot \\left(-\\frac{t}{1-t}\\right) > 0 $$ $$ p\_c(r') > (1-p\_c(r')) \\frac{t}{1-t} $$ $$ p\_c(r')(1-t) > (1-p\_c(r'))t $$ $$ p\_c(r') - p\_c(r')t > t - p\_c(r')t $$ $$ p\_c(r') > t $$
    
*   **结论**: 在带惩罚的新评分机制下，理性代理人的最优策略是：仅当其对某个答案的内在置信度 $p\_c(r')$ 超过了外部设定的风险阈值 $t$ 时，才给出该答案。这从根本上改变了激励结构，使诚实表达不确定性成为一种理性行为。
    

**总结**

论文首先通过一个创新的理论框架，证明了幻觉是统计学习在面对知识内在复杂性时的必然产物，其错误率下界可由分类问题的难度和训练数据的统计特性（如独有样本率）来量化。随后，论文通过博弈论分析和社会实证考察，揭示了现行评估范式在激励层面如何系统性地加剧了这一问题。最终，论文提出的“显式置信度目标”方案，通过重塑评分规则，为引导AI走向更诚实、更可信赖的未来，提供了一个清晰、可行的、基于社会技术协同的路径图。