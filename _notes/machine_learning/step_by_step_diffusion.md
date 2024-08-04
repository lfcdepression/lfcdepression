---
title: "Step-by-Step Diffusion: An Elementary Tutorial"
collection: notes
permalink: /notes/Elementary_Diffusion
date: 2024-08-04

---

## 目录

- [目录](#目录)
- [1 扩散模型基础](#1-扩散模型基础)
  - [1.1 高斯噪声的添加](#11-高斯噪声的添加)
  - [1.2 抽象的扩散](#12-抽象的扩散)
  - [1.3 离散化](#13-离散化)
- [2 随机采样：DDPM](#2-随机采样ddpm)
  - [2.1 DDPM 的正确性](#21-ddpm-的正确性)
  - [2.2 算法](#22-算法)
  - [2.3 方差缩减，预测x0](#23-方差缩减预测x0)
  - [2.4 扩散模型作为SDE](#24-扩散模型作为sde)
- [3 确定性采样：DDIM](#3-确定性采样ddim)
  - [3.1 Case 1: 单点](#31-case-1-单点)
  - [3.2 速度场和气体](#32-速度场和气体)
  - [3.3 Case 2: 双点](#33-case-2-双点)
  - [3.4 Case 3: 任意分布](#34-case-3-任意分布)
  - [3.5 概率流 ODE](#35-概率流-ode)
  - [3.6 讨论：DDPM vs DDIM](#36-讨论ddpm-vs-ddim)
  - [3.7 关于泛化的备注](#37-关于泛化的备注)
- [4 Flow Matching](#4-flow-matching)
  - [4.1 Flow](#41-flow)
  - [4.2 点态流](#42-点态流)
  - [4.3 边缘流](#43-边缘流)
  - [4.4 点对点流的简单选择](#44-点对点流的简单选择)
  - [4.5 Flow Matching](#45-flow-matching)
- [参考文献](#参考文献)

## 1 扩散模型基础

是生成模型的目标是：给定来自某些未知分布 $p(x)$ 的独立分布样本，构建一个（大致）相同分布的采样器。例如通过给定基础分布 $p_{dog}$ 的狗图像训练集，我们希望可以有一种方法生成新的狗图像。

一种思路是学习某个易于采样的分布（例如高斯噪声）到目标 $p$ 的变换。扩散模型的基本思路类似于一滴墨水滴入清水中的扩散及其逆过程，对于一滴墨水难以采样，但是扩散达到平衡态后形成均一的溶液，使采样变得简单。

### 1.1 高斯噪声的添加

对于高斯扩散，设 $x_0$ 是在 $\mathbb{R}^d$ 中按目标分布 $p$ 分布的随机变量。然后通过连续添加具有某些小尺度 $\sigma$ 的独立高斯噪声构造一系列随机变量 $x_1,x_2,\cdots,x_T$​:

$$\begin{equation}
    x_{t+1}:=x_t+\eta_t,\quad\eta_t\sim\mathcal{N}(0,\sigma^2)
\end{equation}$$

这被称为正向（forward）扩散过程，这个过程可以将给定数据分布转换为高斯噪声。方程(1)定义了一个关于所有${x_t}$的联合分布，并且我们令$\{p_t\}_{t\in[T]}$表示每个$x_t$的边缘分布（边缘分布：将联合概率分布投影到其组成随机变量之一的概率分布）。当步骤数$T$较大时，分布$p_T$与高斯分布接近，可以通过采样一个高斯分布来近似采样$p_T$。


![高斯扩散过程](machine_learning/diffusion_elimentary/p1.png)

现在我们可以考虑将问题转换成，给定一个$p_T$如何求得$p_{T-1}$，我们将其称为反向采样器(reverse sampler)，假设我们有一个可用的反向采样器，我们可以从$p_T$即简单的高斯分布开始，迭代应用，得到$p_{T-1},p_{T-2},\cdots p_0$。扩散模型的关键之处在于学习每个中间过程的反向过程比直接从目标分布中采样更容易，有许多种反向采样器的构建方法，我们下面以DDPM为例。

理想的 DDPM 采样器使用了一个简单的策略：在时间 $t$，给定输入$z$（从 $p_t$ 中采样的样本），我们输出从如下条件分布中采样的样本：

$$\begin{equation}
    p(x_{t-1}\vert  x_t=z)
\end{equation}$$

这显然是一个正确的反向采样器。问题在于，它需要为每个 $x_t$ 学习条件分布 $p(x_{t−1} \vert  x_t)$ 的生成模型，这可能会很复杂。但是，如果每步的噪声 $\sigma$ 足够小，那么这个条件分布会变得简单：

**Fact1：(Diffusion Reverse Process)** 对于小的 $\sigma$ 和在（1）中定义的高斯扩散过程，条件分布 $p(x_{t−1} \vert  x_t)$  本身接近高斯分布。也就是说，对于所有时间 $t$ 和条件 $z ∈ \mathbb{R}^d$，存在一些均值参数 $\mu \in \mathbb{R}^d$  使得：

$$\begin{equation}
    p(x_{t-1}\vert  x_{t}=z)\approx\mathcal{N}(x_{t-1}; \mu , \sigma^{2})
\end{equation}$$

这是一个nontrivial的结果，我们将在后面证明它，这个结果带来了一个重大的简化：我们不在需要从头开始学习任意分布$p(x_{t−1} \vert  x_t)$，我们现在对于这个分布除了均值都已经了解了，我们使用$\mu_{t-1}(x_t)$表示其均值（均值$μ_{t−1} : \mathbb{R}^d \rightarrow \mathbb{R}^d$，因为$p(x_{t−1} \vert  x_t)$的均值取决于时间$t$和条件$x_t$）。当$\sigma$足够小时，我们可以将后验分布近似为高斯分布，因此**只要得到条件分布的均值，就可以完整学习这个条件分布**。

![高斯反向过程](machine_learning/diffusion_elimentary/p2.png)

学习 $p(x_{t−1} \vert  x_t)$ 的均值要比学习完整的条件分布简单得多，可以通过回归方法解决。具体而言，我们有一个联合分布 $(x_{t−1}, x_t)$，我们很容易可以从中采样，并且我们希望估计 $E[x_{t−1} \vert  x_t]$。这可以通过优化标准的回归损失来实现：

$$\begin{equation}
    \begin{aligned}
\mu_{t-1}(z)& :=\mathbb{E}[x_{t-1}\vert  x_t=z] \\
\Longrightarrow\mu_{t-1}& =\underset{f:\mathbb{R}^d\to\mathbb{R}^d}{\operatorname*{argmin}}\quad\underset{x_t,x_{t-1}}{\operatorname*{E}}\vert \vert f(x_t)-x_{t-1}\vert \vert _2^2 \\
&=\underset{f:\mathbb{R}^d\to\mathbb{R}^d}{\operatorname*{argmin}}\underset{x_{t-1},\eta}{\operatorname*{\mathbb{E}}}\vert \vert f(x_{t-1}+\eta_t)-x_{t-1}\vert \vert _2^2,
\end{aligned}
\end{equation}$$

其中，期望是针对目标分布$p$中采样得到的样本$x_0$来说的，这个特定的回归问题在某些情况下已经被广泛研究。当目标$p$是图像分布时，对应的回归问题实际上是图像去噪的目标，可以使用CNN等方法处理。**我们已经将从任意分布中抽样学习的问题简化为了标准的回归问题**。

### 1.2 抽象的扩散

抽象地讨论高斯扩散之外的情况，定义类似的扩散模型，以便了解多种方法包括确定性采样器(deterministic samplers)，离散域(discrete domains)和流匹配(flow-matching)等。抽象来讲，构建类似的扩散生成模型的方法如下：从目标分布$p$开始，选择一些容易抽样的基础分布$q(x)$，例如高斯分布等，然后试图构建一些列在目标分布$p$和基础分布$q$之间的插值分布序列，也就是说，我们构建分布：

$$\begin{equation}
    p_0 , p_1 , p_2 , \ldots,p_T,
\end{equation}$$

$p_0=p$ 作为我们的目标分布，$p_T=q$ 作为基分布，并且相邻的分布 $(p_{t−1}, p_t)$ 在某种程度上，边际上是“接近的”，之后我们学习一个反向采样器，将分布$p_t$转换为$p_{t-1}$。如果相邻分布足够接近，即$\sigma$足够小，这一步骤会更加容易。

**Definition 1 (Reverse Sampler)** 给定一系列边际分布 $p_t$，第 $t$ 步的反向采样器是一个潜在的随机函数 $F_t$​，满足如果 $x_t∼p_t$，那么 $F_t(x_t)$ 的边际分布恰好是 $p_{t−1}$:

$$\begin{equation}
    \{F_t(z):z\sim p_t\}\equiv p_{t-1}
\end{equation}$$

存在很多可能得反向采样器，这些采样器并不依赖于高斯噪声，甚至不需要“添加噪声”的概念，我们可以在离散设置中将其实例化，其中我们考虑在有限集合上的分布$p$，并定义相应的插值分布和反向采样器。甚至我们可以构建具有确定性的反向采样器，在后面的部分我们将看到三种流行的采样器，DDPM，DDIM（确定性采样器）以及流匹配（可以视为DDIM的一般化）对于给定的一组边际分布$p_t$，存在许多可能得联合分布于这些边际分布一致。**因此对于给定的一组边际分布，没有一个标准的反向采样器**。

### 1.3 离散化

我们在进一步展开之前需要更精确说明相邻分布的“接近”概念，我们希望将序列$p_0,p_1,\cdots p_T$视为某种时间演化函数$p(x,t)$的离散化，该函数从$t=0$与目标分布$p_0$开始，到时间$t=1$与噪声分布$p_T$结束：

$$\begin{equation}
    p(x,k\Delta t)=p_k(x),\quad\mathrm{where~}\Delta t=\frac1T
\end{equation}$$

步数$T$控制离散化的精细度（相邻分布的“接近”程度），如果$T$足够大可以考虑连续时间极限。为了保证最终分布$p_T$的方差与离散化的步数无关，我们还需要具体的确定每个增量的方差：如果$x_k=x_{k-1}+\mathcal{N}(0,\sigma^2)$，那么，$x_T\sim\mathcal{N}(x_0,T\sigma^2)$。因此，我们需要通过$\Delta t=1/T$来放缩方差：

$$\begin{equation}
    \sigma=\sigma_q\sqrt{\Delta t}
\end{equation}$$

其中，$σ^2_q$是所需要的终端方差（terminal variance），这个选择确保了无论$T$如何选择，$p_T$的方差始终为$σ^2_q$。这种$\sqrt{\Delta t}$放缩在之后对反向求解器正确性的论证中将被证明是重要的，并且与SDE表述相关。

为了方便我们将调整符号，从现在开始，$t$将表示在区间$[0,1]$内的连续值（具体来说，取值为$0,\Delta t,2\Delta t,\cdots,T\Delta t=1$），下标将表示时间而非索引，例如$x_t$现在表示在离散时间$t$的$x$，方程（1）变为：

$$\begin{equation}
    x_{t+\Delta t}:=x_t+\eta_t,\quad\eta_t\sim\mathcal{N}(0,\sigma_q^2\Delta t)
\end{equation}$$

因此有：

$$\begin{equation}
    x_t\sim\mathcal{N}(x_0,\sigma_t^2),\quad\mathrm{where~}\sigma_t:=\sigma_q\sqrt{t}
\end{equation}$$

因为到时间 $t$ 时添加的总噪声：$\sum_{\tau\in\{0,\Delta t,2\Delta t,...,t-\Delta t\}}\eta_\tau $，是一个均值为$0$，方差为 $\sum_\tau\sigma_q^2\Delta t=\sigma_q^2t$ 的高斯分布。

## 2 随机采样：DDPM

在这一节中，回顾了在前面讨论的类似DDPM的反向采样器，并证明其正确性。这个采样器在概念上与Ho 等人（2020年）在去噪扩散概率模型（DDPM）中推广的采样器相同，最初由 Sohl-Dickstein 等人（2015年）引入。相较于Ho等人的工作，这一节中的技术细节略有不同，主要区别在于使用了“方差爆炸（Variance Exploding）” 的扩散前向过程，还使用了一个恒定的噪声计划，并且没有讨论如何参数化预测器（“预测 $x_0 \text{~vs.~} x_{t−1} \text{~vs.~} \text{noise~} \eta$”）

我们考虑1.3中的设计，其中有一个目标分布$p$，以及由方程定义的噪声联合分布$(x_0, x_{\Delta t},...,x_1​)$,DDPM采样器需要估计下面的条件期望：

$$\begin{equation}
    \mu_t(z):=\mathbb{E}[x_t\vert  x_{t+\Delta t}=z]
\end{equation}$$

 对于每个时间步 $t\in{0,\Delta t,...,1−\Delta t}$都有一个$\{μ_t\}$。在训练阶段，我们通过优化如下去噪回归目标来从 $x_0 $的独立同分布样本估计这些函数：

$$\begin{equation}
    \mu_t=\underset{f:\mathbb{R}^d\to\mathbb{R}^d}{\operatorname*{argmin}}\underset{x_t,x_{t+\Delta t}}{\operatorname*{\mathbb{E}}}\vert \vert f(x_{t+\Delta t})-x_t\vert \vert _2^2
\end{equation}$$

通常使用参数化$f$的神经网络，在使用中，通常会共享参数来学习不同的回归函数$\{μ_t\}$，而不是独立为每个时间步学习单独的函数，通常训练一个模型$f_{\theta}$，该模型接受时间$t$作为额外参数，并且满足$f_{\theta}(x_t,t)=\mu_t(x_t)$来实现。在推理阶段，我们使用估计的函数来实现反向采样器：

**Algorithm 1 Stochastic Reverse Sampler (DDPM-like)**: 对于输入样本和时间$x_t,t$，有

$$\widehat{x}_{t-\Delta t}\leftarrow\mu_{t-\Delta t}(x_t)+\mathcal{N}(0,\sigma_q^2\Delta t)$$

实际生成样本需要从各向同性的高斯分布中采样$x_1\sim\mathcal{N}(0,\sigma_q^2)$，然后运行 **Algorithm 1** 迭代直到$t=0$以生成样本$x_0$

我们想推理这个过程的正确性：为什么迭代算法**Algorithm 1**可以近似生成一个来自目标分布$p$的样本？我们需要证明某种形式的**Fact 1**，即真实条件分布$p(x_{t-\Delta t}\vert  x_t)$可以很好近似为高斯分布，并且随着$\Delta t \rightarrow 0$这种近似会更加准确。

### 2.1 DDPM 的正确性

对**Fact1**进行更精确的推导，完成对**Algorithm 1**的正确性的论证——即它在**Definition 1** 的意义下近似于一个有效的反向抽样器。

**Claim 1** (Informal) 设 $p_{t−Δt}$(x) 是任意充分平滑的 $\mathbb{R}^d$ 上的密度。考虑 $(x_{t−Δt}, x_t)$ 的联合分布，其中$x_{t-\Delta t}~\sim~p_{t-\Delta t}$且$x_t~\sim~x_{t-\Delta t}+\mathcal{N}(0,\sigma_q^2\Delta t)$。那么，对于足够小的 $\Delta t$，以下成立。对于所有的条件 $z\in \mathbb{R}^d$，存在 $\mu_z$ 使得：

$$\begin{equation}
    p(x_{t-\Delta t}\vert  x_{t}=z)\approx\mathcal{N}(x_{t-\Delta t}; \mu_{z} , \sigma_{q}^{2}\Delta t)
\end{equation}$$

其中 $\mu_z$ 是仅依赖于 $z$ 的某个常数。此外，仅需

$$\begin{equation}
    \begin{aligned}\mu_{z}&:=\mathbb{E}_{(x_{t-\Delta t},x_t)}[x_{t-\Delta t}\vert  x_t=z]\\&=z+(\sigma_q^2\Delta t)\nabla\log p_t(z),\end{aligned}
\end{equation}$$

其中，$p_t$ 是 $x_t$ 的边际分布。

在推导之前，需要主义几点：**Claim 1**意味着，要从$x_{t−Δt}$ 抽样，只需首先从$x_t$ 抽样，然后从以 $\mathbb{E}[x_{t-\Delta t}\vert  x_t]$为中心的高斯分布抽样。这正是 DDPM所做的。我们实际上不需要方程中的$\mu_z$ 的表达式；我们只需知道这样的 $\mu_z$存在，因此我们可以从样本中学习它。

**Claim 1**(Informal)的证明：这里是为什么分数出现在反向过程中的启发性论证。我们基本上只需应用贝叶斯规则，然后适当地进行泰勒展开。我们从贝叶斯规则开始：

$$\begin{equation}
    p(x_{t-\Delta t}\vert x_t)=p(x_t\vert x_{t-\Delta t})p_{t-\Delta t}(x_{t-\Delta t})/p_t(x_t)
\end{equation}$$ 

然后对两边取对数。在整个过程中，我们会忽略对数中的任何加法常数（这些对应于归一化因子），并且放弃所有 $\mathcal{O}(\Delta t)$ 阶的项。注意，在这个推导中，我们应将 $x_t$ 视为常数，因为我们希望理解条件概率作为 $x_{t−Δt}$ 的函数。

$$\begin{equation}
    \begin{aligned}
&\log p(x_{t-\Delta t}\vert x_t)=\log p(x_t\vert x_{t-\Delta t})+\log p_{t-\Delta t}(x_{t-\Delta t})-\log p_t(x_t) \\
&=\log p(x_t\vert x_{t-\Delta t})+\log p_t(x_{t-\Delta t})+\mathcal{O}(\Delta t) \\
&=-\frac1{2\sigma_q^2\Delta t}\vert \vert x_{t-\Delta t}-x_t\vert \vert _2^2+\log p_t(x_{t-\Delta t}) \\
&=-\frac1{2\sigma_q^2\Delta t}\vert \vert x_{t-\Delta t}-x_t\vert \vert _2^2 \\
&+\log p_t(x_t)+\langle\nabla_x\log p_t(x_t),(x_{t-\Delta t}-x_t)\rangle+\mathcal{O}(\Delta t) \\
&=-\frac1{2\sigma_q^2\Delta t}\left(\vert \vert x_{t-\Delta t}-x_t\vert \vert _2^2-2\sigma_q^2\Delta t\langle\nabla_x\log p_t(x_t),(x_{t-\Delta t}-x_t)\rangle\right) \\
&=-\frac1{2\sigma_q^2\Delta t}\vert \vert x_{t-\Delta t}-x_t-\sigma_q^2\Delta t \nabla_x\log p_t(x_t)\vert \vert _2^2+C \\
&=-\frac1{2\sigma_q^2\Delta t}\vert \vert x_{t-\Delta t}-\mu\vert \vert _2^2
\end{aligned}
\end{equation}$$

这与具有均值 $\mu$ 和方差 $\sigma_{q}^{2}\Delta t$ 的正态分布的对数密度相同，差别仅在于加法因子。因此

$$\begin{equation}
    p(x_{t-\Delta t}\vert  x_t)\approx\mathcal{N}(x_{t-\Delta t}; \mu,\sigma_q^2\Delta t)
\end{equation}$$

思考这个推导，其主要思想是对于足够小的 $\Delta t$，反向过程 $p(x_{t-Δt} \vert x_t)$ 的贝叶斯规则展开由正向过程中的项 $p(x_t \vert  x_{t-Δt})$ 主导。这就是为什么反向过程和正向过程具有相同的函数形式的原因。并且，这种正向和反向过程之间的一般关系，比仅限于高斯扩散更为普遍。

**Technical Details [Optional]：** **Claim 1**不足以证明DDPM算法的正确性，问题在于随着$\Delta t$的减小，我们每步近似的误差减小，但需要的总步数增加。因此如果每步的误差减小的程度小于总步数增加增大的程度的话，误差会积累到一个很大的程度。因此我们需要更进一步的量化。下面的**Lemma 1**表明如果每步噪声的方差为$\sigma^2$，则每步高斯近似的KL误差为$\mathcal{O}(\sigma^4)$.这种衰减速度相当快，因为步数随着$\Omega(1/\sigma^2)$增长。

**Lemma 1**设 $p(x)$ 是 $\mathbb{R}$ 上的任意密度，具有有界的1到4阶导数。考虑联合分布 $(x_0, x_1)$，其中 $x_0 ∼ p，x_1 ∼ x_0 + \mathcal{N}(0, \sigma^2)$。那么，对于任意的条件 $z\in \mathbb{R}$，我们有 

$$\begin{equation}
    \mathrm{KL}\left(\mathcal{N}(\mu_z,\sigma^2) \vert \vert  p_{x_0\vert x_1}(\cdot \vert  x_1=z)\right)\leq O(\sigma^4)
\end{equation}$$

其中：

$$\begin{equation}
    \mu_z:=z+\sigma^2\nabla\log p(z)
\end{equation}$$

这个结论可以通过泰勒展开来证明。

### 2.2 算法

**Pseudocode 1 2**  给出了显式的 DDPM 训练损失和采样代码。为了训练网络 $f_{\theta}$。训练过程通过均匀从区间[0,1]中抽样$t$来同时优化所有时间步$t$的$f_{\theta}$，通常通过反向传播来最小化**Pseudocode 1**输出的期望损失$L_{\theta}$。**Pseudocode 3**描述了密切相关的DDIM采样器。

![DDPM算法](machine_learning/diffusion_elimentary/p3.png)

### 2.3 方差缩减，预测x0

到目前为止，我们的扩散模型已经被训练来预测 $\mathbb{E}[x_{t-\Delta t}\vert  x_t]$：这是**Algorithm 1** 所要求的，也是**Pseudocode 1**训练过程产生的。然而，许多实际的扩散实现实际上是训练来预测$\mathbb{E}[x_{0}\vert  x_t]$，预测初始点的期望，而不是前一个点$x_{t-\Delta t}$，这种差异是一个方差缩减的技巧，在期望上估计相同的量，在形式上有如下关联：

**Claim 2** 对于高斯扩散设置，我们有：

$$\begin{equation}
\mathbb{E}[(x_{t-\Delta t}-x_t)\vert  x_t]=\frac{\Delta t}t\mathbb{E}[(x_0-x_t)\vert  x_t]    
\end{equation}$$

或者等价的：

$$\begin{equation}
    \mathbb{E}[x_{t-\Delta t}\vert  x_t]=\left(\frac{\Delta t}{t}\right)\mathbb{E}[x_0\vert  x_t]+\left(1-\frac{\Delta t}{t}\right)x_t
\end{equation}$$

**Claim 2**暗示，如果我们想要估计$\mathbb{E}[x_{t-\Delta t}\vert x_t]$，我们可以改为估计$\mathbb{E}[x_0\vert x_t]$，然后除以目前的步数即 $(t/\Delta t)$ ，DDPM 训练和抽样算法的方差缩减版本即使用这种方法。

![方差缩减图示](machine_learning/diffusion_elimentary/p4.png)

**Claim 2**背后的原理可以在上图中得到解释：首先，给定$x_t$预测$x_{t-\Delta t}$相当于预测最后的噪声步，即在正向过程中的$\eta_{t-\Delta t}=(x_t-x_{t-\Delta t})$。但如果只有最后的$x_t$，那么之前所有的噪声步$\{\eta_i\}_{i<t}$在直觉上“看起来是相同的”，我们无法区分最后一步添加的噪声和之前的噪声。通过这种对称性，我们可以推断，所有单独的噪声步在给定$x_t$的条件下是相同分布的（尽管不是独立的）。因此，我们可以不估计单个噪声步，而是等效估计所有之前噪声步的平均值，这样具有更低的方差。在到达时间$t$时有$t/\Delta t$ 个过去的噪声步，因此我们将总噪声除以这个数来计算平均值。

扩散模型始终被训练来估计期望。特别是我们训练模型预测$\mathbb{E}[x_0\vert x_t]$时，我们不应将其视为学习“如何从分布$p(x_0\vert x_t)$中采样”。例如我们在训练一个图像扩散模型，那么最优模型输出类似图像的模糊混合物的$\mathbb{E}[x_0\vert x_t]$——它看起来不像实际的图像样本，扩散模型论文中讨论模型“预测$x_0$”并不意味生成类似于实际样本的东西。

### 2.4 扩散模型作为SDE

在本节中，我们将迄今讨论的离散时间过程与随机微分方程（SDE）联系起来。在连续极限下，当$\Delta t\to 0$ 我们的离散扩散过程变成一个随机微分方程。SDE 还可以表示许多其他扩散变体（对应于不同的漂移（drift）和扩散项），在设计选择上提供了灵活性，比如缩放和噪声调度。SDE 视角非常强大，因为现有理论为时间反转 SDE 提供了一般的闭式解。我们特定扩散的反向时间 SDE 的离散化立即得出了我们在本节中推导的采样器，但其他扩散变体的反向时间 SDE 也可以自动获得（然后可以用任何现成的或定制的 SDE 求解器来解决），从而实现更好的训练和抽样策略，我们将在后面进一步讨论。

**The Limiting SDE**

回顾离散更新的方法：

$$\begin{equation}
    x_{t+\Delta t}=x_t+\sigma_q\sqrt{\Delta t}\xi,\quad\xi\sim\mathcal{N}(0,1)
\end{equation}$$

在 $\Delta t\to 0$ 的极限下，这对应于一个零漂移的 SDE：

$$\begin{equation}
    dx=\sigma_qdw
\end{equation}$$

其中$w$是布朗运动（Brownian motion），可以在其他非平衡统计物理或随机过程的书中找到Brownian motion和Itô 公式的详细内容，布朗运动是一个方差随$\Delta t$缩放的具有独立同分布高斯增量的随机过程。具有启发性的我们考虑$dw \sim \lim_{\Delta t\to0}\sqrt{\Delta t}\mathcal{N}(0,1)$，并有下面的公式推导出（23）：

$$\begin{equation}
    dx=\lim_{\Delta t\to0}(x_{t+\Delta t}-x_t)=\sigma_q\lim_{\Delta t\to0}\sqrt{\Delta t}\xi=\sigma_qdw
\end{equation}$$

更普遍地，不同的扩散变体等价于具有不同漂移和扩散项选择的 SDE：

$$\begin{equation}
    dx=f(x,t)dt+g(t)dw
\end{equation}$$

SDE（23）只是$f=0,g=\sigma_q$的一种特殊情况，对应于对$f,g$的不同选择，这种形式涵盖了很多其他可能性。这种灵活性对于新算法开发相当重要。实际中做出的两个重要选择是调整噪声和缩放$x_t$，他们有助于控制$x_t$的方差，并控制我们对于不同噪声水平的关注程度，可以采用灵活的噪声$\{\sigma_t\}$来替代固定的$\sigma\equiv \sigma_q\cdot\sqrt{t}$：

$$x_t\sim\mathcal{N}(x_0,\sigma_t^2)\iff x_t=x_{t-\Delta t}+\sqrt{\sigma_t^2-\sigma_{t-\Delta t}^2z_{t-\Delta t}}\iff dx=\sqrt{\frac d{dt}\sigma^2(t)}dw$$

如果我们还希望通过因子 $s(t)$ 缩放每个 $x_t$ ，Karras 等人 [2022] 表明这对应于以下 SDE：

$$x_t\sim\mathcal{N}(s(t)x_0,s(t)^2\sigma(t)^2)\iff f(x)=\frac{\dot{s}(t)}{s(t)}x,\quad g(t)=s(t)\sqrt{2\dot{\sigma}(t)\sigma(t)}$$

这些只是灵活的SDE所支持的丰富而有用的设计空间的几个例子。

**Reverse-Time SDE**

反向时间 SDE 是对像 DDPM 这样的采样器的连续时间类比。Anderson [1982] 提出的一个深刻结果（在 Winkler [2021] 中重新推导得很好）表明，SDE 的时间反转由以下公式给出

$$\begin{equation}
    dx=\left(f(x,t)-g(t)^2\nabla_x\log p_t(x)\right)dt+g(t)d\overline{w}
\end{equation}$$


也就是说，SDE (26) 告诉我们如何将任何形式为 (25) 的 SDE 在时间上反向运行！这意味着我们不必在每种情况下重新推导反向公式，可以选择任何 SDE 求解器来获得实际的采样器。 **But nothing is free:** 我们仍然不能直接使用 (25) 进行反向采样，因为项 $\nabla_x \log ⁡p_t (x)$ ——事实上是之前出现在方程 14 中的分数——通常是未知的，因为它取决于 $p_t$。然而，如果我们能够学习到这个分数，那么我们就可以求解反向 SDE。这类似于离散扩散，其中正向过程易于建模（它只是添加噪声），而反向过程必须被学习。

分数$∇_x \log ⁡p_t (x)$，起着核心作用。直观地，由于分数 “指向较高概率”，它有助于反转扩散过程，而扩散过程 “平滑” 了概率。分数还与给定 $x_t$ 条件下 $x_0$ 的期望有关。回想一下，在离散情况下，根据方程 14 和 20：

$$\begin{equation}
    \sigma_q^2\Delta t\nabla\log p_t(x_t)=\mathbb{E}[x_{t-\Delta t}-x_t\vert  x_t]=\frac{\Delta t}t\operatorname{E}[x_0-x_t\vert  x_t]
\end{equation}$$

相似地，在连续情况下，我们有：

$$\begin{equation}
    \sigma_q^2\nabla\log p_t(x_t)=\frac1t\operatorname{E}[x_0-x_t\vert x_t]
\end{equation}$$

我们可以通过直接应用 Tweedie 公式来看到这一点，该公式表明：

$$\begin{equation}
    \mathbb{E}[\mu_z\vert z]=z+\sigma_z^2\nabla\log p(z)\mathrm{~for~}z\sim\mathcal{N}(\mu_z,\sigma_z^2)
\end{equation}$$

由于 $x_t∼\mathcal{N}(x_0, t·\sigma_q^2)$，应用 Tweedie 公式，设 $z \equiv x_t，\mu_z ≡ x_0$​，得到：

$$\begin{equation}
    \mathbb{E}[x_0\vert x_t]=x_t+t\sigma_q^2\nabla\log p(x_t)
\end{equation}$$

回到反向 SDE，我们可以证明其离散化获得了**Claim 1** 中的 DDPM 采样器作为一个特例。简单 SDE（23） 的反转为：

$$\begin{equation}
    \begin{aligned}dx&=-\sigma_q^2\nabla_x\log p_t(x)dt+\sigma_qd\overline{w}\\&=-\frac1t\operatorname{E}[x_0-x_t\vert  x_t]dt+\sigma_qd\overline{w}\end{aligned}
\end{equation}$$

其离散化为：

$$\begin{equation}
    \begin{aligned}
x_t-x_{t-\Delta t}& =-\frac{\Delta t}t\operatorname{IE}[x_0-x_t\vert  x_t]+\mathcal{N}(0,\sigma_q^2\Delta t) \\
&=-\operatorname{E}[x_{t-\Delta t}-x_t\vert  x_t]+\mathcal{N}(0,\sigma_q^2\Delta t) \\
\Longrightarrow x_{t-\Delta t}& =\mathbb{E}[x_{t-\Delta t}\vert  x_t]+\mathcal{N}(0,\sigma_q^2\Delta t) 
\end{aligned}
\end{equation}$$

这正是**Claim 1**中推导的随机（DDPM）采样器。

## 3 确定性采样：DDIM

**Algorithm 2 Deterministic Reverse Sampler (DDIM-like)**: 对于输入样本和时间$x_t,t$，有

$$\begin{equation}
    \widehat{x}_{t-\Delta t}\leftarrow x_t+\lambda(\mu_{t-\Delta t}(x_t)-x_t)
\end{equation}$$

其中$\lambda:=\left(\frac{\sigma_t}{\sigma_{t-\Delta t}+\sigma_t}\right)$ 且 $\sigma_t\equiv \sigma_q\sqrt{t}$

我们如何证明，这定义了一个有效的反向采样器？由于**Algorithm 2** 是确定性的，表明其从 $p(x_{t−Δt} \vert x_t)$ 进行采样是没有意义的，因为我们对类似 DDPM 的随机采样器进行了这样的论证。相反，我们将直接展示方程（33）实现了边际分布 $p_t$ 和 $p_{t−Δt}$ 之间的有效转移映射（transport map）。也就是说，如果我们让 $F_t$ 是方程（33）的更新： 

$$\begin{equation}
    \begin{aligned}F_{t}(z)&:=z+\lambda(\mu_{t-\Delta t}(z)-z)\\&=z+\lambda(\mathbb{E}[x_{t-\Delta t}\vert  x_t=z]-z)\end{aligned}
\end{equation}$$

那么我们想要证明

$$\begin{equation}
    F_t\sharp p_t\approx p_{t-\Delta t}
\end{equation}$$

其中符号 $F_t\sharp p$ 表示 $\{F(x)\}_{x∼p}$ 的分布。这被称为 $p$ 经过函数 $F$的推前分布（pushforward）

**证明概述** : 通常证明这一点的方法是使用随机微积分的工具。但我们的策略是首先在最简单的情况下，即点质量分布的情况下展示**Algorithm 2** 是正确的，然后通过适当地进行边缘化将这一结果推广到完整的分布情况，这类似于 “流匹配” 证明。

### 3.1 Case 1: 单点

让我们首先理解目标分布 $p_0$ 是 $\mathbb{R}^d$ 中的单点质量分布的简单情况。因为我们可以通过简单地“平移”我们的坐标来达到这个目的。从形式上讲，我们的整个设置，包括方程 34，都是平移对称的，我们可以假设该点位于 $x_0 = 0$。在这种情况下，**Algorithm 2** 是否正确？

为了推理正确性，我们希望考虑任意步骤 $t$ 时 $x_t$ 和 $x_{t−Δt}$ 的分布。根据扩散前向过程，在时间 $t$时，相关的随机变量是：

$$\begin{equation}
\begin{aligned}
\chi_{0}& =0\quad(\text{deterministically}) \\
x_{t-\Delta t}& \sim\mathcal{N}(x_0,\sigma_{t-\Delta t}^2) \\
x_{t}& \sim\mathcal{N}(x_{t-\Delta t},\sigma_t^2-\sigma_{t-\Delta t}^2). 
\end{aligned}    
\end{equation}$$

我们在这些协方差中省略单位矩阵，以简化符号，可以假设维度 $d = 1$

$x_{t−Δt}$ 的边缘分布是$p_{t-\Delta t}=\mathcal{N}(0,\sigma_{t-1}^2)$，而 $x_t$ 的边缘分布是$p_t=\mathcal{N}(0,\sigma_t^2)$。

让我们首先找到一些确定性函数 $G_t:\mathbb{R}^d\to\mathbb{R}^d$，使得 $G_t\sharp p_t=p_{t-\Delta t}$。有许多可能的函数可以起作用。例如，我们总是可以将原点周围的旋转添加到任何有效的映射中，但有一个明显的选择： 

$$\begin{equation}
    G_t(z):=\left(\frac{\sigma_{t-\Delta t}}{\sigma_t}\right)z
\end{equation}$$

函数 $G_t$ 简单地重新调整了 $p_t$ 的高斯分布，以匹配 $p_{t−Δt}$ 的高斯分布的方差。事实证明，这个 $G_t$ 正好等同于**Algorithm 2** 中采取的步骤 $F_t$，我们现在将展示这一点。

**Claim 3** 当目标分布是一个点质量 $p0 = \delta_0$ 时，更新步骤 $F_t$等价于缩放函数 $G_t$： 

$$\begin{equation}
    F_t\equiv G_t
\end{equation}$$

因此，**Algorithm 2** 定义了目标分布 $p0 = \delta_0$ 的逆采样器。

*Proof：*

为了应用 $F_t$，我们需要计算我们简单分布的 $\mathbb{E}[x_{t-\Delta t}\vert x_t]$。由于 $(x_{t-\Delta t},x_t)$ 是联合高斯分布，即

$$\begin{equation}
    \mathbb{E}[x_{t-\Delta t}\vert  x_{t}]=\left(\frac{\sigma_{t-\Delta t}^{2}}{\sigma_{t}^{2}}\right)x_{t}
\end{equation}$$

剩下的是代数运算：

$$\begin{aligned}
F_t(x_t)& :=x_t+\lambda(\mathbb{E}[x_{t-\Delta t}\vert  x_t]-x_t) \\
&=x_t+\left(\frac{\sigma_t}{\sigma_{t-\Delta t}+\sigma_t}\right)(\mathbb{E}[x_{t-\Delta t}\vert  x_t]-x_t) \\
&=x_{t}+\left(\frac{\sigma_{t}}{\sigma_{t-\Delta t}+\sigma_{t}}\right)\left(\frac{\sigma_{t-\Delta t}^{2}}{\sigma_{t}^{2}}-1\right)x_{t} \\
&=\left(\frac{\sigma_{t-\Delta t}}{\sigma_t}\right)x_t \\
&=G_t(x_t).
\end{aligned}$$

因此，我们得出结论，**Algorithm 2**  是一个正确的逆采样器，因为它等价于 $G_t$，并且 $G_t$ 是有效的。

如果 $x_0$ 是任意点而不是 $x_0=0$，**Algorithm 2** 的正确性仍然成立，因为整个设置是平移对称的。

### 3.2 速度场和气体

将 DDIM 视为等价于速度场可能会有所帮助，该速度场将时间 $t$ 的点移动到时间 $(t - \Delta t)$ 的位置。具体来说，定义向量场：

$$\begin{equation}
    v_t(x_t):=\frac\lambda{\Delta t}(\mathbb{E}[x_{t-\Delta t}\vert  x_t]-x_t)
\end{equation}$$

DDIM 更新算法可以写为

$$\begin{equation}
\begin{aligned}
    \widehat{x}_{t-\Delta t}:&=x_t+\lambda(\mu_{t-\Delta t}(x_t)-x_t)\\&=x_t+v_t(x_t)\Delta t.    
\end{aligned}
\end{equation}$$

对于速度场 $v_t$ 的物理直觉是：想象一个非相互作用的粒子气体，其密度场由 $p_t$ 给出。然后，假设位置为 $z$ 的粒子沿着速度场 $v_t(z)$ 移动。结果气体的密度场将是 $p_{t−Δt}$。我们将这一过程表示为：

$$\begin{equation}
    p_t\xrightarrow{v_t}p_{t-\Delta t}
\end{equation}$$

在步长 $\Delta t$ 很小的极限情况下，非正式地说，我们可以将 $v_t$ 视为速度场——它指定了按照 DDIM 算法移动的粒子的即时速度。

作为一个具体的例子，如果目标分布 $p_0=\delta_{x0​​}$，如第 3.1 节所述，则 DDIM 的速度场是$v_t(x_t)=\left(\frac{\sigma_t-\sigma_{t-\Delta t}}{\sigma_t}\right)(x_0-x_t)/\Delta t$这是一个指向初始点 $x_0$ 的向量场（参考下图）

![单点速度场](machine_learning/diffusion_elimentary/p5.png)

### 3.3 Case 2: 双点

现在让我们展示当目标分布是两个点的混合时，**Algorithm 2** 是正确的：

$$\begin{equation}
    p_0:=\frac12\delta_a+\frac12\delta_b
\end{equation}$$

其中 $a,b\in\mathbb{R}^d$。根据扩散前向过程，在时间 $t$ 的分布将是高斯混合体，在这里，前向过程的线性性（关于 $p_0$ 的）是很重要的。粗略地说，扩散一个分布相当于独立地扩散该分布中的每个个体点；这些点之间没有相互作用：

$$\begin{equation}
    p_t:=\frac{1}{2}\mathcal{N}(a,\sigma_t^2)+\frac{1}{2}\mathcal{N}(b,\sigma_t^2)
\end{equation}$$

我们希望展示在这些分布 $p_t$ 下，DDIM 的速度场 $v_t$ 可以转移$p_t\xrightarrow{v_t}p_{t-\Delta t}$。

让我们首先尝试构造某些速度场 $v_t^*$，使得 $p_t$ 经 $v_t^*$ 转移到 $p_{t−Δt}$。根据我们在第 3.1 节的结果——即 DDIM 更新对单点有效——我们已经知道可以转移每个混合成分 $\{a, b\}$ 的速度场。也就是说，我们知道速度场 $v_t^{[a]}$ 定义为

$$\begin{equation}
    v_t^{[a]}(x_t):=\lambda\underset{x_0\sim\delta_a}{\mathbb{E}}[x_{t-\Delta t}-x_t\vert  x_t]
\end{equation}$$

请特别注意我们对哪些分布取期望！方程式（45）中的期望是针对单点分布 $\delta a$ 的，但是我们对 DDIM 算法的定义及其方程式（40）中的向量场总是针对目标分布而言的。在我们的情况下，目标分布是方程式（43）中的 $p_0$

$$\begin{equation}
    \mathcal{N}(a,\sigma_t^2)\xrightarrow{v_t^{[a]}}\mathcal{N}(a,\sigma_{t-\Delta t}^2)
\end{equation}$$

与 $v_t^{[b]}$ 相似。

现在我们希望找到一种方法将这两个速度场组合成一个单一的速度场 $v_{t}^{*}$，以转移这个混合分布：

$$\begin{equation}
    \underbrace{\left(\frac12\mathcal{N}(a,\sigma_t^2)+\frac12\mathcal{N}(b,\sigma_t^2)\right)}_{p_t}\xrightarrow{v_t^*}\underbrace{\left(\frac12\mathcal{N}(a,\sigma_{t-\Delta t}^2)+\frac12\mathcal{N}(b,\sigma_{t-\Delta t}^2)\right)}_{p_{t-\Delta t}}
\end{equation}$$

我们可能会尝试简单地取平均速度场 $v_{t}^{*}= 0.5v^{[a]}_t + 0.5v^{[b]}_t$ ​，但这不正确。 $v_{t}^{*}$ 是各个单独速度场的加权平均，其权重由它们对应的密度场确定。 

$$\begin{equation}
    \begin{aligned}
v_{t}^{*}(x_{t})& =\frac{v_t^{[a]}(x_t)\cdot p(x_t\vert  x_0=a)+v_t^{[b]}(x_t)\cdot p(x_t\vert  x_0=b)}{p(x_t\vert  x_0=a)+p(x_t\vert  x_0=b)} \\
&=v_{t}^{[a]}(x_{t})\cdot p(x_{0}=a\vert  x_{t})+v_{t}^{[b]}(x_{t})\cdot p(x_{0}=b\vert  x_{t})
\end{aligned}
\end{equation}$$

明确地说，在点 $x_t$ 处，速度场 $v^{[a]}_t$ 的权重是 $x_t$ 是由初始点 $x_0 = a$ 生成的概率，而不是 $x_0 = b$ 的概率。

为了直观地理解这一点，考虑图中所示的关于气体的对应问题。假设我们有两种重叠的气体：红色气体密度为$\mathcal{N}(a,\sigma^2)$，速度为 $v^{[a]}_t$；蓝色气体密度为 N(b, σ^2)，速度为 $v^{[b]}_t$。我们想知道，合并气体的有效速度是多少（就像我们只看到灰度图一样）。显然，我们应该通过它们各自的密度加权平均个别气体的速度，就像在方程（48）中一样。

![双点速度场](machine_learning/diffusion_elimentary/p6.png)

现在我们已经解决了本节的主要子问题：我们找到了一个特定的向量场 $v_t^*$，它将 $p_t$ 转移到 $p_{t−\Delta t}$ ，适用于两点分布 $p_0$。现在剩下的是展示这个 $v_t^*$ 是否等价于 **Algorithm 2** 的速度场。

为了展示这一点，首先注意到单个向量场 $v_t^{[a]}$ 可以写成一个条件期望。利用方程式（45）中的定义，我们添加条件 $x_0=a$ ，因为我们希望根据两点混合分布，而不是单点分布来取期望

$$\begin{equation}
    \begin{aligned}
v_t^{[a]}(x_t)& =\lambda\underset{x_0\sim\delta_a}{\operatorname*{\mathbb{E}}}[x_{t-\Delta t}-x_t\vert  x_t] \\
&=\lambda\underset{x_0\sim1/2\delta_a+1/2\delta_b}{\operatorname*{\operatorname*{\mathbb{E}}}}[x_{t-\Delta t}-x_t\vert  x_0=a,x_t]
\end{aligned}
\end{equation}$$

现在整个向量场 $v^*_t$ 可以写成一个条件期望：

$$\begin{equation}
    \begin{aligned}
v_t^*(x_t)& =v_t^{[a]}(x_t)\cdot p(x_0=a\vert  x_t)+v_t^{[b]}(x_t)\cdot p(x_0=b\vert  x_t) \\
&=\lambda\mathbb{E}[x_{t-\Delta t}-x_t\vert  x_0=a,x_t]\cdot p(x_0=a\vert  x_t) \\
&+\lambda\mathbb{E}[x_{t-\Delta t}-x_t\vert  x_0=b,x_t]\cdot p(x_0=b\vert  x_t) \\
&=\lambda\mathbb{E}\left[x_{t-\Delta t}-x_t\vert  x_t\right] \\
&=v_t(x_t)
\end{aligned}
\end{equation}$$

其中，所有的期望都是相对于分布 $x_{0} \sim 1/2\delta_{a}+1/2\delta_{b}$。因此，组合的速度场 $v^*_t$ 正是由 **Algorithm 2** 更新给出的速度场 $v_t$ ——因此 **Algorithm 2** 对我们的两点混合分布是一个正确的逆采样器。

### 3.4 Case 3: 任意分布

在3.3中了解了如何处理两个点的情况，我们可以将这种思想推广到$x_0$的任意分布。**Algorithm 2**的整体证明策略可以推广到其他类型的扩散过程，而没有太多区别，这引出了流匹配的概念，在后面部分，介绍了流的机制，直接从简单的单点缩放推导DDIM是容易的。

### 3.5 概率流 ODE

最后，我们将我们的离散时间确定性采样器推广到称为概率流 ODE 的常微分方程（ODE）[Song et al., 2020]。以下部分建立在我们在第 2.4 节中讨论的将 SDE 作为扩散的连续极限的基础上。正如第 2.4 节中逆时间 SDE 提供了离散随机采样器的灵活连续时间推广一样，我们将看到离散确定性采样器如何推广到 ODE。

ODE 形式提供了一个有用的理论视角，用于理解扩散，同时具有实际优势，如选择各种现成和定制 ODE 求解器来改善采样（例如流行的 DPM++ 方法）。

回顾第 2.4 节中的一般 SDE：

$$\begin{equation}
    dx=f(x,t)dt+g(t)dw
\end{equation}$$

Song 等人 [2020] 展示了可以将这个 SDE 转换为一个称为概率流 ODE（PF-ODE）的确定性等效形式：

$$\begin{equation}
    \frac{dx}{dt}=\tilde{f}(x,t),\quad\mathrm{where~}\tilde{f}(x,t)=f(x,t)-\frac12g(t)^2\nabla_x\log p_t(x)
\end{equation}$$

SDE和ODE在下面的意义上是等价的：通过求解PF-ODE得到的轨迹有与SDE轨迹相同的边缘分布。使用气体类比来说SDE描述了气体中个别粒子的运动，而PF-ODE描述了气体流速场的流线，也就是PF-ODE描述了被气体转移的“测试粒子”的运动。然而类似于在反向SDE中一样，我们也需要学习得分来应用ODE。

DDPM是反向时间SDE的特例，DDIM也可以被视为PF-ODE的特例，回顾之前$f=0,g=\sigma_q$的SDE，相应的ODE是：

$$\begin{equation}
    \begin{aligned}
\frac{dx}{dt}& =-\frac12\sigma_q^2\nabla_x\log p_t(x) \\
&=-\frac1{2t}\operatorname{E}[x_0-x_t\vert  x_t]
\end{aligned}
\end{equation}$$

反向和离散化的结果如下：

$$\begin{equation}
    \begin{aligned}
x_{t-\Delta t}& =x_t+\frac{\Delta t}{2t}\mathbb{E}[x_0-x_t\vert  x_t] \\
&=x_t+\frac12(\mathbb{E}[x_{t-\Delta t}\vert  x_t]-x_t)
\end{aligned}
\end{equation}$$

注意到$\lim_{\Delta t\to0}\left(\frac{\sigma_t}{\sigma_{t-\Delta t}+\sigma_t}\right) = \frac12$，我们恢复了DDIM的采样器。

### 3.6 讨论：DDPM vs DDIM

DDPM和DDIM在概念上有所区别，一种是随机的，而另一种是确定性的。

- DDPM理想情况下实现了一个随机映射$F_t$，有输出的$F_t(x_t)$在每个点上都是从条件分布 $p(x_{t−Δt}\vert x_t)$ 中采样得到的。
  
- DDIM 理想情况下实现了一个确定性映射 $F_t$，使得输出 $F_t(x_t)$ 的边际分布为 $p_{t−Δt}$。i.e. $F_t\sharp p_t=p_{t-\Delta t}$

尽管在相同输入$x_t$的情况下向相同的方向进行，但这两种方法最终的演变方式非常不同。让我们考虑理想情况下的采样器的行为，从初始点$x_1$开始并且迭代完成。

- DDPm类型情况下从$p(x_0\vert  x_1)$采样得到样本，如果正向过程充分混合$\sigma_q$很大，那么终点$x_1$将几乎独立于初始点。因此$p(x_0\vert  x_1) \approx p(x_0)$，理想的DDPM输出完全不依赖于起始点$x_1$，相反DDIM是确定性的，因此对于固定的$x_1$会产生固定值，依赖于$x_1$。
  
- 留下的一个图像是，DDIM定义了一个确定性映射$\mathbb{R}^d\to \mathbb{R}^d$，从一个高斯分布采样到目标分布。在这个层面上而言，DDIM映射类似于其他生成模型（GANs和Normalizing Flows也定义了从高斯噪声到真实分布的映射）。DDIM的特殊性在于它不是任意的：目标分布$p$完全确定了理想DDIM映射，训练的模型是为了模拟这个映射，这个映射是“好的”，如果我们的目标是光滑的，我们也希望它是平滑的。相较而言GANs可以自由学习噪声和图像之间的任意映射，扩散模型这个特性可能会在某些情况下更容易学习（因为是监督学习），也可能在某些情况下使学习更困难（因为可能有更容易学习的映射）。

### 3.7 关于泛化的备注

我们没有讨论在只有有限样本和有限资源的情况下，如何学习底层分布的属性，这是扩散模型的基本方面，目前还没有被完全理解。

为了欣赏这里的微妙之处，假设我们使用经验风险最小化（Empirical Risk Minimization，ERM）的经典策略来学习扩散模型：我们从底层分布中采样一个有限的训练集，并针对这个经验分布优化所有的回归函数。问题在于，我们不应该完全最小化经验风险，因为这会导致一个只能复现训练样本的扩散模型。

通常情况下，学习扩散模型必须进行正则化，无论是隐式地还是显式地，以防止过拟合和训练数据的记忆。当我们训练用于扩散模型的深度神经网络时，这种正则化通常是隐式的：有限的模型大小和优化随机性等因素阻止了训练模型完全记忆其训练集。

这个记忆训练数据的问题已经在小图像数据集上的扩散模型中被观察到，并且已经观察到，随着训练集大小的增加，记忆效应减少了 [Somepalli et al., 2023, Gu et al., 2023]。此外，像 Carlini 等人 [2023] 中指出的那样，记忆还被视为神经网络潜在的安全和版权问题，作者发现他们可以从稳定的扩散中恢复训练数据。下图展示了训练集大小的影响，并展示了使用 3 层 ReLU 网络训练的扩散模型的 DDIM 轨迹。我们可以看到，$N = 10$ 样本的扩散模型 “记忆” 了其训练集：其轨迹都收敛到训练点中的一个，而不是产生底层的螺旋分布。随着样本的增加，模型开始泛化：轨迹收敛到底层的螺旋流形。这些轨迹还开始更垂直于底层流形，表明正在学习低维结构。我们还注意到，在 $N = 10$ 的情况下，扩散模型失败了，人类可能无法从这些样本中识别出 “正确” 的图样（pattern），所以泛化的期望可能太高了。

![DDIM轨迹](machine_learning/diffusion_elimentary/p7.png)

## 4 Flow Matching

现在介绍流匹配（flow matching）的框架 [Peluchetti, 2022, Liu et al., 2022b,a, Lipman et al., 2023, Albergo et al., 2023]。流匹配可以被看作是 DDIM 的一种推广，它在设计生成模型时提供了更大的灵活性，包括例如稳定扩散 [Liu et al., 2022a, Esser et al., 2024] 中使用的修正流（有时称为线性流）。实际上，在我们对 DDIM 进行的分析中，我们已经看到了流匹配的主要思想。在高层次上，这里是我们在第 3 节构建生成模型的方式：

- 首先，我们定义了如何生成单个点。具体来说，我们构建了矢量场 $\{v_t^{[a]}\}$，当应用于所有时间步长时，将标准高斯分布转换为任意的 delta 分布 $\delta_a$。
  
- 其次，我们确定了如何将两个矢量场组合成单个有效的矢量场。这使我们能够构建从标准高斯到两个点（或者更一般地，到点的分布—— 即目标分布）的转移。

这两个步骤都不特别需要高斯基分布，或者高斯前向过程。例如，将两个矢量场组合的第二步对于任意两个矢量场来说仍然是相同的。

因此，可以放弃高斯假设，我们可以从基本的层面考虑如何在任意两点之间进行映射，然后我们可以研究两个点分别从任意分布$p$（数据）和$q$（基本）中采样会发生什么。这种观点涵盖了DDIM作为其中的一个特例。

### 4.1 Flow

让我们首先定义流（flow）的核心概念。流是一组时间索引的矢量场 $v = \{v_t\}，t\in [0,1]$。可以将这看作是在每个时间 $t$ 上的气体的速度场 $v_t$，正如在第 3.2 节中所做的那样。任何流通过沿着速度场 $\{v_t\}$ 将初始点 $x_1$ 转移到最终点 $x_0$，定义了一个轨迹。

形式化地，对于流 $v$ 和初始点 $x_1$，考虑 ODE：

$$\begin{equation}
    \frac{dx_t}{dt}=-v_t(x_t)
\end{equation}$$

相应的离散时间模拟是迭代过程： 在 $t = 1$ 时从初始点 $x_1$ 开始，通过 $x_{t-Δt} \leftarrow x_t + v_t(x_t) \Delta t$ 进行迭代

在时间 $t = 1$ 时，初始条件为 $x_1$。我们写作：

$$\begin{equation}
    x_t:=\operatorname{RunFlow}(v,x_1,t)
\end{equation}$$

来表示在时间 $t$ 结束时（到达最终点 $x_0$），流 ODE的解。换句话说，RunFlow 是沿着流 $v$ 将点 $x_1$ 转移到时间 $t$ 的结果。

正如流定义了初始点和最终点之间的映射，它们也通过沿着它们的轨迹“推进”源分布中的点，定义了整个分布之间的传输。如果 $p_1$ 是初始点的分布。大多数流匹配文献使用反向时间约定，因此 $t = 1$ 是目标分布。为了与 DDPM 约定保持一致，我们将 $t = 0$ 视为目标分布，则应用流 $v$ 将得到最终点的分布：

$$\begin{equation}
    p_0=\{\text{RunFlow}(v,x_1,t=0)\}_{x_1\sim p_1}
\end{equation}$$

我们用$p_1\stackrel{v}{\hookrightarrow}p_0$表示表示流 $v$ 将初始分布 $p_1$ 转移到最终分布 $p_0$ 的过程。在我们的气体类比中，这意味着如果我们从按照 $p_1$ 分布的粒子气体开始，并且每个粒子按照流 $v$ 定义的轨迹运动，那么最终粒子的分布将是 $p_0$。

流匹配的最终目标是以某种方式学习一个流 $v^*$，它将易于采样的基础分布 $q$（例如高斯分布）转移到目标分布 $p$ ，即$q\stackrel{v^*}{\hookrightarrow}p$，如果我们有了这个 $v^*$,我们可以通过首先从$q$中采样$x_1$，然后运行流，以 $x_1$ 作为初始点，并输出结果的最终点 $x_0$，从而生成目标分布 $p$ 的样本。DDIM 算法实际上是这种情况的一个特例（DDIM的连续时间极限是一个流，其中$v_t(x_t)=\frac1{2t}\mathbb{E}[x_0-x_t\vert x_t]$基础分布 $p_1$ 是高斯分布。DDIM 抽样 **Algorithm 3** 是评估 RunFlow 的离散化方法。DDPM 训练（**Algorithm 2**）是学习 v* 的方法，但它依赖于高斯结构，并且在某些方面与我们将在本章中呈现的流匹配算法有所不同。）针对流 $v^*$ 的一个非常特定的选择。那么，我们如何一般构造这样的流呢？

### 4.2 点态流

我们的基本构建模块将是点态流（pointwise flow），它仅将单个点 $x_1$ 转移到点 $x_0$。直观地说，给定连接 $x_1$ 和 $x_0$ 的任意路径 $\{x_t\},t\in [0,1]$，点态流通过在每个点 $x_t$ 处给出其速度 $v_t(x_t)$ 来描述这条轨迹。形式上，$x_1$ 和 $x_0$ 之间的点态流是任何满足方程 55，并在 $t = 1$ 和 $t = 0$ 时具有边界条件 $x_1$ 和 $x_0$ 的流 $\{v_t\}_t$。我们将这样的流表示为 $v^{[x_1,x_0]}$。点态流并不唯一：在 $x_0$ 和 $x_1$ 之间存在许多不同的路径选择。

![点态流](machine_learning/diffusion_elimentary/p8.png)

### 4.3 边缘流

对于所有点对$(x_1, x_0)$都可以构建一个显式的点态流$v^{[x1,x0]}$，将点源$x_1$转移到目标点$x_0$。例如，可以让$x_t$沿着从$x_1$到$x_0$的任意显式路径移动。回想气体类比，这相当于一个例子在$x_1$和$x_0$之间移动。现在让我们建立一组粒子，使得在$t=1$时，粒子按照$q$分布，在 $t = 0$ 时，按照$p$分布，这是容易做到的：可以选择任意一个$q$与$p$之间的耦合$\Pi_{q,p}$，并且考虑与点态流$\{v^{[x_1,x_0]}\}_{(x_1,x_0)\sim\Pi_{q,p}}$ 对应的粒子，给出一个点态流的分布（即一组粒子的轨迹），在总体上表现为我们需要的。

我们希望以某种方式结合所有的点态流，得到一个单一的流$v^*$，实现相同的分布之间的转移，这样可以简化学习问题，不需要学习所有个体轨迹的分布，只需要学习它们整体演变的速度场。

为了确定有效速度 $v_t^*(x_t)$，我们应该通过所有个体粒子速度 $v_{t}^{[x_{1},x_{0}]}$ 的加权平均值来确定，其中权重是在 $x_t$ 处的粒子由点级流 $v^{[x_1,x_0]}$ 生成的概率。最终结果

$$\begin{equation}
    v_t^*(x_t):=\underset{x_0,x_1\vert x_t}{\mathbb{E}}[v_t^{[x_1,x_0]}(x_t)\vert  x_t]
\end{equation}$$

其中，期望对应于由采样 $(x_1,x_0)~\sim~\Pi_{q,p}$ 引起的 (x1,x0,xt) 的联合分布，其中$(x_1,x_0,x_t)$，令$x_t\leftarrow\mathrm{RunFlow}(v^{[x_1,x_0]},x_1,t)$。

在高层次上查看这个结果的另一种方式是：我们从它们转移 $\delta$ 分布的点对点流 $v^{[x1,x0]}$ 开始：

$$\begin{equation}
    \delta_{x_1}\stackrel{v[x_1,x_0]}{\hookrightarrow}\delta_{x_0}
\end{equation}$$

然后方程（58）以一种高级方式 “对 $x_1$ 和 $x_0$ 上的这些流进行平均”，得到转移流 $v^*$：

$$\begin{equation}
     q=\underset{x_{1}\sim q}{\operatorname*{\mathbb{E}}}[\delta_{x_{1}}]\stackrel{v^{*}}{\hookrightarrow}\underset{x_{0}\sim p}{\operatorname*{\mathbb{E}}}[\delta_{x_{0}}]=p
\end{equation}$$

从原则上讲，我们对生成建模问题有了一个 “解决方案”，但在实践中仍有一些重要的问题需要解决：

- 我们应该选择哪个点对点流 $v^{[x_1,x_0]}$ 和耦合 $\Pi_{q,p}$
- 我们如何计算边际流 $v^*$ ？我们不能直接从方程（58）计算它，因为这将需要对给定点 $x_t$ 从 $p(x_0\vert x_t)$ 进行采样，这在一般情况下可能很复杂。

### 4.4 点对点流的简单选择

我们需要明确选择：点对点流、基础分布 $q$ 和耦合 $\Pi_{q,p}$。有许多简单的可行选择，基础分布 $q$ 可以是任何易于采样的分布。高斯分布是一个常见的选择，但绝非唯一的选择。下图使用了一个环形的基础分布，基础分布和目标分布之间的耦合$\Pi_{q,p}$，最简单的选择是独立耦合，即从$p$和$q$中独立采样。

![点对点流](machine_learning/diffusion_elimentary/p9.png)

![边流](machine_learning/diffusion_elimentary/p10.png)

对于点对点流而言，可以说最简单的构造是线性点对点流：

$$\begin{equation}
    v_t^{[x_1,x_0]}(x_t)=x_0-x_1,\\\Longrightarrow\text{ RunFlow}(v^{[x_1,x_0]},x_1,t)=tx_1+(1-t)x_0
\end{equation}$$

这简单地在线性上对 $x_1$ 和 $x_0$ 进行插值（并对应于 Liu 等人 [2022a] 中的选择）。上图可视化了由线性点对点流组成的边际流，与环形基础分布 $q$ 相同，目标分布为点质量 $(p=\delta_{x_0})$。

### 4.5 Flow Matching

现在，唯一剩下的问题是，直接用方程（58）评估 $v^*$ 需要针对给定的 $x_t$ 从 $p(x_0\vert  x_t)$ 进行采样。如果我们知道如何在 $t=1$ 时执行此操作，我们就已经解决了生成建模问题。

幸运的是，可以利用与 DDPM 相同的技巧：只需能够从联合分布 $(x_0,x_t)$ 进行采样，然后解决一个回归问题。类似于 DDPM：

$$\begin{equation}
    \begin{aligned}
v_{t}^{*}(x_{t})& :=\underset{x_0,x_1\vert x_t}{\operatorname*{\mathbb{E}}}[v_t^{[x_1,x_0]}(x_t)\vert  x_t] \\
\Longrightarrow v_t^*& =\underset{f:\mathbb{R}^d\to\mathbb{R}^d}{\mathbb{E}}\underset{(x_0,x_1,x_t)}{\operatorname*{E}}\vert \vert f(x_t)-v_t^{[x_1,x_0]}(x_t)\vert \vert _2^2 
\end{aligned}
\end{equation}$$

上面的式子表明，为了计算模型$f_{\theta}$在固定时间$t$的损失，应该：

- 从联合分布中采样源点和目标点$(x_1,x_0)$

- 通过点对点流$v^{[x_1,x_0]}$，从点$x_1$到时间$t$确定性计算$x_t$。如果使用线性的那么我们有$x_{t}\leftarrow tx_{1}+(1-t)x_{0}$。

- 在$x_t$处评估模型的预测$f_{\theta}(x_r)$，评估确定性向量$v_t^{[x_1,x_0]}(x_t)$，然后计算两个量之间的L2损失。


为了从训练好的模型（即对 $v_{t}^{*}$ 的估计）中采样，我们首先从源点 $x_{1}\sim q$ 中采样，然后沿着学到的流将其传输到目标样本 $x_0$。**Pseudocode 4 5** 给出了基于流的模型训练和采样的明确过程（包括具体情况下的线性流）。

![伪代码45](machine_learning/diffusion_elimentary/p11.png)

**总结**

下面是如何为目标分布 $p$ 学习流匹配生成模型的方法。

*The Ingredients*

我们首先选择：

- 一个源分布 $q$，我们有效从中采样（例如标准高斯分布）。

- 源分布 $q$ 和目标分布 $p$ 之间的耦合 $\Pi_{q,p}$​，它指定了如何联合采样一对源点和目标点 $(x_1,x_0)$，分别具有边际分布 $q$ 和 $p$。标准选择是独立耦合，即独立地从 $q$ 中采样 $x_1$ 和从 $p$ 中采样 $x_0$。
  
- 对所有点对 $(x_1,x_0)$，显式点对点流 $v^{[x_1,x_0]}$ 将 $x_1$ 转移到 $x_0$。我们必须能够在所有点上有效地计算向量场 $v_t^{[x_1,x_0]}$。

这些成分理论上决定了边际向量场 $v^*$，它将 $q$ 转移到 $p$：

$$\begin{equation}
    v_t^*(x_t):=\underset{x_0,x_1\vert x_t}{\mathbb{E}}[v_t^{[x_1,x_0]}(x_t)\vert  x_t]
\end{equation}$$

其中期望对应于联合分布：

$$\begin{equation}
    \begin{aligned}(x_1,x_0)&\sim\Pi_{q,p}\\x_t&:=\text{RunFlow}(v^{[x_1,x_0]},x_1,t)\end{aligned}
\end{equation}$$

**训练**

通过反向传播 **Pseudocode 4** 计算的随机损失函数来训练神经网络 $f_{\theta}$，对于这个期望损失的最优函数是：$f_\theta(x_t,t)=v_t^*(x_t)$

**采样**

运行 **Pseudocode 5**，从（近似）目标分布 $p$ 中生成一个样本 $x_0$。


## 参考文献

[1] 1. Nakkiran, P., Bradley, A., Zhou, H. & Advani, M. Step-by-Step Diffusion: An Elementary Tutorial.  Preprint at https://doi.org/10.48550/arXiv.2406.08929 (2024).

[2] [（2024，DDPM，DDIM，流匹配，SDE，ODE）扩散：基础教程](https://blog.csdn.net/qq_44681809/article/details/140063268)