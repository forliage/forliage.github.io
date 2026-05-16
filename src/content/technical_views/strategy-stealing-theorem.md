---
title: "Strategy Stealing Theorem"
description: "A proof for the strategy stealing theorem"
pubDate: 2026-05-16
tags: ["Strategy", "Combinational Analysis", "Existing Problem"]
heroImage: "/images/Gomoku.png"
---

## I. Formal Model: Symmetric Positional Games

Let $X$ be a finite set, called the **board**. For example, in Gomoku one may take

$$
X=\{1,\dots,m\}\times \{1,\dots,n\}.
$$

Let $\mathcal W\subseteq 2^X$ be a nonempty family of subsets of $X$, called the family of **winning sets**. Each $W\in\mathcal W$ represents a winning pattern. In Gomoku, for instance, $\mathcal W$ may be the family of all sets consisting of five consecutive collinear grid points.

There are two players, denoted by $P_1$ and $P_2$. The rules are as follows.

1. Initially, no point of $X$ is occupied.
2. $P_1$ moves first, and then the two players alternate turns.
3. On each turn, the player chooses one previously unoccupied point of $X$.
4. At any time, let $A_t\subseteq X$ be the set of points occupied by $P_1$, and let $B_t\subseteq X$ be the set of points occupied by $P_2$.
5. If at some time there exists $W\in\mathcal W$ such that $W\subseteq A_t,$ then $P_1$ wins.
6. If at some time there exists $W\in\mathcal W$ such that $W\subseteq B_t,$ then $P_2$ wins.

7. If the whole board is filled and neither player has occupied a winning set, the game is a draw.

This model has two essential properties.

First, the two players have the same winning condition: each player wins by occupying some set $W\in\mathcal W$. Therefore the game is **symmetric**.

Second, the winning condition is monotone: if a set $A\subseteq X$ is already winning, meaning that there exists $W\in\mathcal W$ with $W\subseteq A$, then every larger set $A'\supseteq A$ is also winning.

Formally,

$$
\exists W\in\mathcal W,\ W\subseteq A
\quad\Longrightarrow\quad
\forall A'\supseteq A,\ \exists W\in\mathcal W,\ W\subseteq A'.
$$

This is the mathematical expression of the idea that “having an extra stone of one’s own cannot hurt.”

## II. Formal Definition of Strategies

A position can be written as a triple $(A,B,i),$ where $A\subseteq X$ is the set of points occupied by $P_1$, $B\subseteq X$ is the set of points occupied by $P_2$, $A\cap B=\varnothing,$ and $i\in\{1,2\}$ indicates whose turn it is.

A **strategy** for a player is a function that chooses a legal move based on the current history or current position.

For precision, define a legal history as

$$
h=(x_1,x_2,\dots,x_t),
$$

where each $x_j\in X$, and

$$
x_j\neq x_\ell\qquad (j\neq \ell).
$$

If $t$ is even, it is $P_1$'s turn to move. If $t$ is odd, it is $P_2$'s turn to move.

A strategy for $P_1$ is a function

$$
\sigma_1:\{h:\ |h|\ \text{is even}\}\to X
$$

such that

$$
\sigma_1(h)\notin \{x_1,\dots,x_t\}.
$$

A strategy for $P_2$ is a function

$$
\sigma_2:\{h:\ |h|\ \text{is odd}\}\to X
$$

such that

$$
\sigma_2(h)\notin \{x_1,\dots,x_t\}.
$$

If a strategy for $P_i$ guarantees that $P_i$ eventually wins no matter how the opponent plays, then it is called a **winning strategy** for $P_i$.


## III. Main Theorem: The Strategy-Stealing Theorem

**Theorem: In a Symmetric Monotone Positional Game, the Second Player Has No Winning Strategy**

Let $X$ be a finite board and let $\mathcal W\subseteq 2^X$ be a family of winning sets. Consider the two-player alternating positional game defined above.

Assume that both players have the same winning family $\mathcal W$, and that the winning condition is monotone.

Then

$$
P_2\ \text{has no winning strategy.}
$$

Furthermore, since this is a finite game of perfect information, it follows that

$$
P_1\ \text{has a strategy that guarantees not losing.}
$$

Equivalently,

$$
P_1\ \text{can guarantee either a win or a draw.}
$$

This is the rigorous meaning of the statement that the first player is guaranteed not to lose.

## IV. Complete Proof

The proof has two main steps.

First, we prove that

$$
P_2\ \text{cannot have a winning strategy.}
$$

Second, using the determinacy of finite perfect-information games, we conclude that

$$
P_1\ \text{can guarantee not losing.}
$$

### Step 1: Proof by Contradiction

Assume, for contradiction, that $P_2$ has a winning strategy. Denote this strategy by $\sigma.$

This means that if $P_2$ follows $\sigma$, then no matter how $P_1$ plays, $P_2$ will eventually occupy some winning set:

$$
\exists W\in\mathcal W,\quad W\subseteq B,
$$

where $B$ is the final set of points occupied by $P_2$.

We will show that if such a strategy $\sigma$ exists, then $P_1$ can “steal” it and turn it into a winning strategy for $P_1$. This will produce a contradiction.

### Step 2: The First Player Makes an Arbitrary Extra Move

Since $X\neq\varnothing$, the first player $P_1$ begins by choosing an arbitrary point $x_0\in X.$

Thus the real set of stones of $P_1$ already contains $x_0.$

After that, $P_1$ pretends to be the second player and follows the supposed second-player winning strategy $\sigma$.

To make this rigorous, we introduce a **virtual game**.

## V. Construction of the Virtual Game

In the real game:

- $A$ denotes the set of points occupied by $P_1$;
- $B$ denotes the set of points occupied by $P_2$.

In the virtual game, we reverse the roles:

- the real stones of $P_2$ are regarded as the stones of the virtual first player;
- the real stones of $P_1$, except for some additional unused stones, are regarded as the stones of the virtual second player.

Formally, at every stage we maintain

$$
A=A'\cup E,
$$

$$
B=B',
$$

where:

- $A'$ is the set of stones of the virtual second player;
- $B'$ is the set of stones of the virtual first player;
- $E\subseteq A$ is the set of “extra stones” belonging to the real $P_1$ but not used in the virtual game.

Initially, $A=\{x_0\},$ $B=\varnothing.$

We set

$$
A'=\varnothing,\qquad B'=\varnothing,\qquad E=\{x_0\}.
$$

Therefore initially,$A=A'\cup E,$ $B=B',$ and $A'\cap B'=\varnothing.$

### Step 3: How the First Player Imitates the Second-Player Strategy

Whenever the real $P_2$ makes a move, the set $B$ gains one point. In the virtual game, this point is interpreted as a move by the virtual first player.

Therefore, in the virtual game, it becomes the virtual second player’s turn.

By assumption, the virtual second player has the winning strategy $\sigma$. Hence $\sigma$ specifies a point, denoted by

$$
y=\sigma(h'),
$$

where $h'$ is the current virtual history.

If $y\notin A\cup B,$

meaning that $y$ is still unoccupied in the real game, then the real $P_1$ plays at $y.$

In the virtual game, we also record $y$ as the move of the virtual second player.

Thus we update

$$
A'\leftarrow A'\cup\{y\},
$$

$$
A\leftarrow A\cup\{y\}.
$$

The extra-stone set $E$ remains unchanged.

### Step 4: What If the Required Point Is Already Occupied by an Extra Stone?

This is the subtle part of the strategy-stealing proof.

It may happen that $y\in E.$

That is, the strategy $\sigma$ asks the virtual second player to play at a point $y$, but in the real game this point is already occupied by $P_1$ as an extra stone.

Since

$$
y\in E\subseteq A,
$$

the real $P_1$ already owns the point that $\sigma$ wants.

However, in the real game $P_1$ still must make an actual new move. Therefore $P_1$ chooses any currently unoccupied point

$$
z\in X\setminus(A\cup B)
$$

and plays at $z$.

In the real game,

$$
A\leftarrow A\cup\{z\}.
$$

In the virtual game, we still record $y$ as the move made by the virtual second player:

$$
A'\leftarrow A'\cup\{y\}.
$$

The newly played real point $z$ is put into the extra-stone set:

$$
E\leftarrow (E\setminus\{y\})\cup\{z\}.
$$

Thus the invariant

$$
A=A'\cup E
$$

is preserved.

Intuitively, the extra stone has simply been transferred from $y$ to $z$.


## VI. The Inductive Invariant

We now prove that throughout the entire construction, the following invariant is maintained:

$$
A=A'\cup E,\qquad B=B',\qquad A'\cap B'=\varnothing.
$$

Also,

$$
E\subseteq A.
$$

The invariant holds initially, as shown above.

Assume it holds before some move. Suppose real $P_2$ plays a point $u$. Since the real game move is legal, $u\notin A\cup B.$

Since $B=B',$ we interpret this as a move of the virtual first player:

$$
B'\leftarrow B'\cup\{u\}.
$$

In the real game, $B\leftarrow B\cup\{u\}.$

Therefore the equality $B=B'$ is preserved.

Now the strategy $\sigma$ of the virtual second player gives a legal virtual move $y$. Since $y$ is legal in the virtual game, $y\notin A'\cup B'.$

However, $y$ may already belong to the extra-stone set $E$ in the real game.

We consider two cases.

### Case 1: $y\notin E$

Since $A=A'\cup E$ and $B=B',$ and since $y\notin A'\cup B'$ and $y\notin E,$ it follows that $y\notin A\cup B.$

Hence the real $P_1$ may legally play at $y$.

Update:

$$
A'\leftarrow A'\cup\{y\},
$$

$$
A\leftarrow A\cup\{y\},
$$

$$
E\leftarrow E.
$$

Thus the relation

$$
A=A'\cup E
$$

is preserved.

### Case 2: $y\in E$

In this case, $y$ is already a real stone of $P_1$, but it has not yet been counted as a virtual stone of the virtual second player.

In the virtual game, we let the virtual second player play at $y$:

$$
A'\leftarrow A'\cup\{y\}.
$$

In the real game, $P_1$ must play a new legal point $z$.

As long as the game has not ended, there exists an unoccupied point

$$
z\in X\setminus(A\cup B).
$$

The real $P_1$ plays at $z$:

$$
A\leftarrow A\cup\{z\}.
$$

Then update the extra-stone set:

$$
E\leftarrow (E\setminus\{y\})\cup\{z\}.
$$

Now the new value of $A'\cup E$ is

$$
(A'\cup\{y\})\cup \bigl((E\setminus\{y\})\cup\{z\}\bigr).
$$

Since before the update we had

$$
A=A'\cup E,
$$

the real updated set is

$$
A_{\mathrm{new}}=A\cup\{z\}.
$$

On the other hand,

$$
(A'\cup\{y\})\cup \bigl((E\setminus\{y\})\cup\{z\}\bigr)=
A'\cup E\cup\{z\}=
A\cup\{z\}=
A_{\mathrm{new}}.
$$

Therefore the invariant is again preserved:

$$
A_{\mathrm{new}}=A'_{\mathrm{new}}\cup E_{\mathrm{new}}.
$$

Thus the simulation can continue until the game ends.

## VII. Why This Implies a Win for the First Player

By the contradictory assumption, $\sigma$ is a winning strategy for the second player.

In the virtual game, the virtual second player always follows $\sigma$. Therefore, regardless of how the virtual first player moves, the virtual second player eventually wins.

Hence there exists some winning set $W\in\mathcal W$ such that $W\subseteq A',$ where $A'$ is the final set occupied by the virtual second player.

But by the invariant, $A'\subseteq A.$ Therefore $W\subseteq A'\subseteq A.$ Thus $W\subseteq A.$

This means that in the real game, the real first player $P_1$ has occupied a complete winning set $W$. Therefore $P_1$ wins the real game.

So from the assumption that $P_2$ has a winning strategy, we have constructed a winning strategy for $P_1$.

This is impossible in a zero-sum win/loss game. More explicitly, if $P_2$ has a winning strategy, then no matter what strategy $P_1$ uses, $P_2$ must win. But we have constructed a strategy for $P_1$ that makes $P_1$ win no matter how $P_2$ plays. This is a contradiction.

Therefore the assumption was false.

Hence

$$
P_2\ \text{has no winning strategy.}
$$

## VIII. From “The Second Player Has No Winning Strategy” to “The First Player Cannot Lose”

So far we have proved only

$$
P_2\ \text{has no winning strategy}.
$$

Now we derive

$$
P_1\ \text{has a non-losing strategy}.
$$

This uses a standard fact: finite perfect-information games are determined. Here one does not need any advanced game theory; backward induction over the finite game tree is enough.

Consider the event “$P_2$ wins.” Since the game tree is finite, every terminal position has exactly one of the following outcomes:

$$
P_1\text{ wins},\quad P_2\text{ wins},\quad \text{draw}.
$$

For every position, define whether it is a position from which $P_2$ can force a win.

The recursive definition is as follows:

- If the current position is terminal and the outcome is a $P_2$ win, then it is $P_2$-winning.
- If the current position is terminal and the outcome is not a $P_2$ win, then it is $P_2$-losing.
- If it is $P_2$'s turn, then the position is $P_2$-winning if there exists at least one legal move leading to a $P_2$-winning position.
- If it is $P_1$'s turn, then the position is $P_2$-winning only if every legal move by $P_1$ leads to a $P_2$-winning position.

This is precisely backward induction on the finite game tree.

Therefore, at the initial position, either

$$
P_2\ \text{can force a win},
$$

or

$$
P_1\ \text{can prevent }P_2\text{ from winning}.
$$

We have already proved that the first alternative is impossible:

$$
P_2\ \text{cannot force a win}.
$$

Therefore the second alternative must hold:

$$
P_1\ \text{can prevent }P_2\text{ from winning}.
$$

If $P_2$ does not win, the terminal outcome can only be $P_1\text{ wins}$ or $\text{draw}.$

Hence

$$
P_1\ \text{can guarantee either a win or a draw}.
$$

Equivalently,

$$
P_1\ \text{cannot be forced to lose}.
$$

This is the rigorous meaning of “the first player is guaranteed not to lose.”

## IX. Application to Gomoku

Now apply the general theorem to Gomoku.

Let the board be

$$
X=\{1,\dots,m\}\times \{1,\dots,n\}.
$$

Let $k=5.$ Define the family of winning sets $\mathcal W$ to be all sets of five consecutive collinear grid points. That is, $W\in\mathcal W$ if and only if there exist a starting point $(a,b)\in X$ and a direction

$$
d\in\{(1,0),(0,1),(1,1),(1,-1)\}
$$

such that

$$
W=\{(a,b),(a,b)+d,(a,b)+2d,(a,b)+3d,(a,b)+4d\},
$$

and

$$
(a,b)+jd\in X,\qquad j=0,1,2,3,4.
$$

The two players alternately occupy empty grid points. A player wins if and only if their occupied set contains some $W\in\mathcal W.$

Therefore ordinary Gomoku satisfies the assumptions of the theorem:

1. The board is finite.
2. The two players have identical rules.
3. The two players have the same family of winning sets.
4. Having an extra stone of one’s own cannot turn a winning position into a losing one.
5. There is no randomness.
6. There is no hidden information.
7. The players move alternately.

Hence, by the strategy-stealing theorem,

$$
\text{In ordinary finite-board Gomoku, the second player has no winning strategy.}
$$

Furthermore,

$$
\text{In ordinary finite-board Gomoku, the first player has a non-losing strategy.}
$$

That is,

$$
\text{The first player can guarantee either a win or a draw.}
$$

This is the rigorous form of the statement that the first player is guaranteed not to lose.

## X. When Can We Further Conclude That the First Player Wins?

The theorem above proves only

$$
\text{the first player cannot lose}.
$$

That is,

$$
P_1\text{ wins or the game is a draw}.
$$

To conclude the stronger statement

$$
P_1\text{ wins},
$$

one needs an additional assumption:

$$
\text{A draw is impossible under optimal play.}
$$

Or, more strongly,

$$
\text{Every completely filled board position contains a winning set for one of the two players.}
$$

If draws are impossible, then “the first player can guarantee a win or a draw” becomes

$$
\text{the first player can guarantee a win}.
$$

Therefore,

$$
\text{First-player non-loss}+\text{impossibility of draws}
\Longrightarrow
\text{first-player win}.
$$

However, in ordinary finite-board Gomoku, the impossibility of draws generally cannot be obtained from the strategy-stealing argument alone. Hence the strategy-stealing argument by itself does not prove that the first player wins.

It proves only

$$
\text{the second player has no winning strategy,}
$$

or equivalently,

$$
\text{the first player has a strategy guaranteeing at least a draw.}
$$