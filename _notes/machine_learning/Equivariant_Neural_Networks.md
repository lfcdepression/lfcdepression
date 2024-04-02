---
title: "Equivariant Graph Neural Networks"
collection: notes
permalink: /notes/EGNNs
date: 2024-04-02

---

一篇综述的学习笔记

# 等变神经网络(Equivariant Graph Neural Networks)

## 背景
主要关于两个基础概念：图神经网络和等变性

### 图神经网络(GNN)

在可以想象的应用中（例如分子体系），考虑一个图：$\mathcal{G}=(\mathcal{V},\mathcal{E})$，其中$\mathcal{V},\mathcal{E}$分别是节点和边，节点$i$被赋予特征$h_i$，我们也可以给连接$i,j$节点的边赋予特征$e_{ij}$。为了描述原子我们不能只具有图拓扑和节点特征，还会有其他的信息，比如每个原子会有一个节点特征$h_i$可以是电荷，质量等，也会赋予一个几何特征例如相空间或者位形空间中的坐标，使用GNN处理这些数据的时候可以将等变性质加入模型中，例如平移、旋转、镜像等对称性。Gilmer在2017年改进了一个开创性的消息传递方案，以将主流GNN统一到一个通用架构中。它迭代地为每个节点（或边）进行消息计算和邻域聚合。一般来说，我们有：

$$\begin{aligned}m_{ij}&=\psi_m\left(h_i,h_j,e_{ij}\right),\\h_i'&=\psi_h\left(\{m_{ij}\}_{j\in\mathcal{N}(i)},h_i\right)\end{aligned}$$

$m_{ij}$表示在边$(i,j)$上的信息传递输出，$\psi_h$和$\psi_m$是带参数的函数，$\mathcal{N}(i)$是$i$节点的近邻节点（默认不包括自环），上面的式子描述了消息的生成，下面的式子描述了消息的聚合和节点的更新。

Eq.(1-2)中描述的一个有趣的属性是GNN有置换等变性，因为$\psi_h$是置换等变的(即node-order equivariance)。现代GNNs实际上就是为了解决这个问题来设计的

使用置换等变性构建的框架是不够的，它们仍欠缺发现几何相似性在3D空间的作用。我们需要结合实际问题，将欧氏空间中的几何等变性纳入我们的神经网络框架。

### 等变性

$\mathcal{X}$和$\mathcal{Y}$分别是输入和输出向量空间，二者都被赋予了一套变换：$G\colon G\times\mathcal{X}\to\mathcal{X}$和$G\times\mathcal{Y}\to\mathcal{Y}$，对于二者之间的变换$\phi:\mathcal{X}\to\mathcal{Y}$，如果有对于输入进行一个变换，输出进行一个相同或者是可以对应的变换，那么我们说这个是等变的。

**等变性(Equivariance)：** 对于$\phi:\mathcal{X}\to\mathcal{Y}$对于所有操作$G$满足如下关系，我们称之为是$G$等变的

$$\phi(\rho_{\mathcal{X}}(g)x)=\rho_{\mathcal{Y}}(g)\phi(x),\forall g\in G$$

$\rho_{\mathcal{X}}$和$\rho_{\mathcal{Y}}$分别是输入和输出空间中的群表示，如果$\rho_{\mathcal{Y}}$是恒等的，那么$\phi$是协变的。

**群(Group)：** 
- $O(n)$ 是$n$维正交群(Orthogonal group), 包括旋转和反射。
- $SO(n)$是特殊正交群(Special Orthogonal group), 包括旋转。
- $E(n)$是$n$维欧氏群，包括旋转、反射和平移。
- $SE(n)$是特殊欧式群，包括旋转和平移。
- 李群的元素是来自一个可微的流形。上述几个群都属于李群的具体例子。

**群表示(Group representation)：** 群的表示是一个可逆的线性映射$\rho(g):G\mapsto\mathcal{V}$同时$\rho(g)\rho(h)=\rho(g{\cdot}h),\forall g,h\in G$

## Geometrically Equivariant GNN

### 几何图

处理不同的问题我们需要考虑不同的东西，例如计算能量时，我们需要考虑旋转不变性，而对于分子动力学模拟则需要考虑输入和输出的等变性，为了更好地区分，在下文中，用粗体表示几何矢量，用普通符号表示非几何量：

$$\begin{gathered}
m_{ij} &=\psi_{m}(\mathbf{x}_{i},\mathbf{x}_{j},h_{i},h_{j},e_{ij}) \\
\mathbf{m}_{ij}&=\psi_\mathbf{m}(\mathbf{x}_i,\mathbf{x}_j,h_i,h_j,e_{ij}) \\
h_{i}^{\prime} &=\psi_h(\{m_{ij}\}_{j\in\mathcal{N}(i)},h_i) \\
\mathbf{x}_i^{\prime} &=\psi_{\mathbf{x}}(\{\mathbf{m}_{ij}\}_{j\in\mathcal{N}(i)},\mathbf{x}_i) 
\end{gathered}$$

其中，$m_{ij}$和$\mathbf{m}_{ij}$分别是通过边$(i,j)$传递的标量和方向输出，$\psi_h$ 和$\psi_\mathbf{x}$分别是用于标量特征和几何矢量特征的消息聚合函数。另外：

- $\psi_m$ 对输入$(\mathbf{x}_i,\mathbf{x}_j)$是G-invariant的
-  $\psi_\mathrm{m}$ 对输入$(\mathbf{x}_i,\mathbf{x}_j)$是G-equivariant的
-  $\psi_\mathrm{x}$对输入$\{\mathbf{m_{ij}}\}$$_{j\in N(i)}$ 和$\mathbf{x_i}$ 是G-equivariant的

![1](machine_learning/Equivariant/image1.png){:height="720px" width="240px"}



图示为旋转情况下几何等变消息传递的图示。生成标量消息和矢量消息，然后进行聚合，从而产生等变更新。平移等变性是容易实现的，因为训练模型基本都是使用相对位置，因此我们主要考虑旋转等操作，根据消息的表示方式，将当前的方法分为三类：不可约表示（irreducible representation），正则表示（regular representation），标量化（scalarization）

### 不可约表示Irreducible Representation

根据表示理论，一个紧群(compact group)的线性表示即为对其不可约表示(缩写为irreps)的直接求和得到一个相似的变换。具体对于$SO(3)$群来说，irreps是$(2l+1)\times(2l+1)$的Wigner-D矩阵$\mathbf{D}^l,l=0,1,\cdots $对$SO(3)$有

$$\rho(g)=\mathbf{Q}^{\mathsf{T}}\left(\underset{l}{\bigoplus}\mathbf{D}^{l}(g)\right)\mathbf{Q}$$

其中，$\mathbf{D}^{l}$是Wigner-D矩阵，$\mathbf{Q}$是正交矩阵表示基的变换，$\bigoplus$为直接求和或者矩阵沿对角线拼接。因此，向量空间被分为了$l$ 个子空间，每一个都被$\mathbf{D}^l$所变换，第$l$ 子空间中的向量被称为 $type$-$l$ 向量。例如，在我们的例子中，标量$h_i$是含有$H$个通道的type-0向量，$\mathbf{x}_i$是type-1向量。这些向量通过张量积$\otimes$ 互相作用，然后通过Wigner-D 矩阵的张量积得到 Clebsch-Gordan (CG) 相关系数$\mathbf{C}^{lk}\in\mathbb{R}^{(2l+1)(2k+1)\times(2l+1)(2k+1)}$, 由CG分解所得：

$$\mathbf{D}^k(g)\otimes \mathbf{D}^l(g)=(\mathbf{C}^{lk})^{\mathrm{T}}\left(\bigoplus_{J=|k-l|}^{k+l}\mathbf{D}^J(g)\right)\mathbf{C}^{lk}$$

构建等变信息传递的最后一个步骤是球谐函数$Y_{Jm}$,其为$SO(3)$等变的基础。有了上述的构建元件， Thomas提出了一个满足SE(3)-等变的TFN层：

$$\begin{aligned}\mathbf{m}_{ij}^l&=\sum_{k\geq0}\mathbf{W}^{lk}(\mathbf{x}_i-\mathbf{x}_j)\mathbf{x}_j^k\\\mathbf{x}_i^{\prime l}&=\omega_{ll}\mathbf{x}_i^l+\sum_{j\in\mathcal{N}_{(i)}}\mathbf{m}_{ij}\end{aligned}$$

其中，$\mathbf{x_i}^{\prime l}\in\mathbb{R}^{2l+1}$表示度为 $l$ 的节点$i$的几何向量，$\mathrm{x_i}\in\mathbb{R}^3$ 为节点坐标，$\omega_{ll}$为自作用权重，filter $\mathbf{W}^{lk}\in\mathbb{R}^{(2l+1)\times(2k+1)}$是旋转可引导的(rotation-steerable),表明对于任意旋转 $r\in SO(3)$,满足$\mathbf{W}^{lk}(\mathbf{D}^1(r)\mathbf{x})=\mathbf{D}^1(r)\mathbf{W}^{lk}(\mathbf{x})(\mathbf{D}^k(r))^{-1}$ 。具体来说：

$$\mathbf{W}^{lk}=\sum_{J=|l-k|}^{l+k}\varphi_J^{lk}(||\mathbf{x}||)\sum_{m=-J}^{J}Y_{Jm}(\mathbf{x}/||\mathbf{x}||)\mathbf{C}_{Jm}^{lk}$$

一系列可学习的半径函数$\varphi_J^{lk}\in\mathbb{R}$,球谐函数$Y_{Jm}\in\mathbb{R}$, CG相关系数$\mathbf{C}_{Jm}^{lk}\in\mathbb{R}^{(2l+1)(2k+1)}$

### 正则表示(Regular Representation)

另一种方法使用正则表示直接寻求在群卷积中获得等变性，其将卷积算子作为群上的函数。处理连续和光滑的群时，群卷积的正数就变得难以处理，一个可行的方法是利用李代数。为了这个目的，Finzi et al.提出了LieConv可以通过Lifting操作将输入映射到群中的元素，然后利用PointConv这一trick完成群卷积的离散化计算

$$\begin{gathered}m_{ij}=\varphi(log(u_i^{-1}u_j))h_j\\h_i^{\prime}=\frac1{|\mathcal{N}(i)|+1}\left(h_i+\sum_{j\in\mathcal{N}(i)}m_{ij}\right)\end{gathered}$$

其中，$u_i\in G$是$\mathbf{x}_i$ 的lift，对数log将每一个群成员映射到李代数 $g$ (向量空间),而$\varphi$为参数化的MLP， 通过除以节点数量来进行归一化，即$\mathcal{N}(i)+1$。很明确LieConv只明确了节点特征$h_i$的更新，同时保持几何向量$x_i$不变，这意味着LieConv具有不变性。

通过相似的想法，LieTransformer使用了自注意力机制来动态得计算卷积核中的权重，以求提升模型的性能。因此等变性可以通过任意的李群或者其离散的子群，故基于正则表示的方法享有很高的灵活性。另一方面，由于离散化和采样，这一方法也在计算复杂度和性能之间达到了平衡。除非我们引入外部哈密顿动力学(Hamiltonian dynamics)来重新定义几何向量(如Finzi et al.)，否则这种方法的一个缺点就是我们很难将其推广到几何向量上来。

### 标量化(Scalarization)
除了群表示方法，许多工作采用一个通用的方法即标量化来建模等变性。通常来说，几何向量首先被转换成不变标量，之后再接几个MLPs来控制其量级，最终添加原先的方向以获得等变性。这个想法最早在SchNet和DimeNet提出，不过只具有不变性。SphereNet进一步在标量化的信息传递中添加了角度和扭转的信息，从而使得具有不变性的网络可以区分手性(chirality)。Radial Field实现了等变性的版本，不过只是在几何向量上进行运算，而没有考虑到节点特征。EGNN进一步以下面的范式更新了这个想法：

$$\begin{gathered}
m_{ij}=\varphi_m(h_i,h_j,||\boldsymbol{x}_i-\boldsymbol{x}_j||,e_{ij}), \\
\boldsymbol{x}_i^{\prime}=\boldsymbol{x}_i+\sum_{j\neq i}(\boldsymbol{x}_i-\boldsymbol{x}_j)\varphi_x(m_{ij}), \\
h_i^{\prime}=\varphi_h(h_i,\sum_{j\in\mathcal{N}(i)}m_{ij}) 
\end{gathered}$$

其中，$\mid\mid x_i-x_j\mid\mid$是几何向量$x_i$和$x_j$的标量化形式；$\varphi_m$、$\varphi_x$和$\varphi_h$都是任意的MLPs。通过设置$m_{ij}=(x_i-x_j)\varphi_x(m_{ij})$, EGNN同时传播了节点特征$h_i$和几何向量$x_i$,以等变的方式直接实现了图结构。这个方法的精髓在于构建不变的信息$m_{ij}$,然后沿着径向方向(radial dirction)重新转换回到等变的输出，与我们计算两个带电粒子的库伦力(重力)类似。

## 参考文献
[1] Han, J., Rong, Y., Xu, T., & Huang, W. (2022). Geometrically Equivariant Graph Neural Networks: A Survey. ArXiv, abs/2202.07230.

[2] [论文笔记：几何等变图神经网络综述](https://gabriel-qin.github.io/2023/06/03/Survey-Geometrically-Equivariant-GNN/)