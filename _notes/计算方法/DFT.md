---
title: "A Preliminary of Density Functional Theory"
collection: notes
permalink: /notes/DFT
date: 2024-03-21

---
DFT理论初步

# 密度泛函理论

求解固体系统的电子结构本质上是求解多体薛定谔方程

$$H\Psi(\mathbf r,\mathbf R)=E\Psi(\mathbf r,\mathbf R)$$

具体可以写为

$$\begin{aligned}
\text{H}& =H_{e}+H_{N}+H_{e-N}  \\
&=-\sum_{i}\frac{\hbar^{2}}{2m_{e}}\nabla_{i}^{2}+\frac{1}{2}\sum_{i\neq j}\frac{e^{2}}{|\mathbf{r}_{i}-\mathbf{r}_{j}|} \\
&-\sum_{I}\frac{\hbar^{2}}{2M_{I}}\nabla_{I}^{2}+\frac{1}{2}\sum_{I\neq J}\frac{Z_{I}Z_{J}e^{2}}{|\mathbf{R}_{I}-\mathbf{R}_{J}|}+\sum_{i,I}\frac{Z_{I}e^{2}}{|\mathbf{r}_{i}-\mathbf{R}_{I}|}
\end{aligned}$$

很明显这个式子是无法求解的，一定需要做很多的简化.

## Born-Oppenheimer 近似

主要是原子核和电子运动的解耦，原子核运动速度远慢于电子，原子核运动之后电子可以马上运动到相应的位置，因此可以认为在平衡位置做振动。同理，当考虑原子核运动时电子可以看作是运动的背景势场。于是我们可以把系统波函数写成原子核波函数和电子波函数的直积：

$$\Psi(\mathbf{r},\mathbf{R})=\chi(\mathbf{R})\psi(\mathbf{r},\mathbf{R})$$

右边第一项是原子核波函数，第二项是电子波函数，进一步由于我们处理电子运动时可以近似认为原子核不动，因此$\mathbf{R}$可以作为参数出现，在讨论中可以略去。因此系统的哈密顿量可以被写为：

$$\begin{aligned}
\text{H}& =H_{Nu}+H_{el}  \\
H_{Nu}& =H_{N}+\sum_{I}V_{e}(\mathbf{R}_{I})  \\
H_{el}& =H_{e}+\sum_{i}V_{e}(\mathbf{r}_{i}) 
\end{aligned}$$

其中$V$表示电子和原子核的相互作用势，在单独考虑其中一者时可以作为外场，此时原子核系统通过牛顿方程处理（类似于处理声子），电子系统通过薛定谔方程处理，此时的薛定谔方程被简化为

$$\left(-\sum_i\frac{\hbar^2}{2m_e}\nabla_i^2+\frac12\sum_{i\neq j}\frac{e^2}{|\mathbf{r}_i-\mathbf{r}_j|}+\sum_iV_e(\mathbf{r}_i)\right)\psi(\mathbf{r},\mathbf{R})=E\psi(\mathbf{r},\mathbf{R})$$

已经简化了很多，但还是不可求解，需要做进一步的近似（我们比较熟悉的的只有单粒子的薛定谔方程）

## Hartree-Fock近似

电子间的相互作用仍然是一个问题，如果采用近自由电子近似可能会缺失很多重要的信息，因此Hartree在此基础上将电子间相互作用视为一种平均势场，提出了Hartree平均场近似，这样多体问题就变成了单体问题，最终的多体波函数变为单体波函数的乘积：

$$\psi(\mathbf{r})=\prod\limits_i\varphi_i(\mathbf{r}_i)$$

我们此时可以通过变分求能量最小值：

$$\delta\left[\langle H\rangle-\sum_iE_i(\langle\varphi_i(\mathbf{r})|\varphi_i(\mathbf{r})\rangle-1)\right]=0$$

体系总能量的期望值对于单体波函数变分得到单电子运动方程：

$$\left(-\frac{\hbar^{2}}{2m_{e}}\nabla^{2}+V(\mathbf{r})+\sum_{i\neq i^{\prime}}\int\mathrm{dr}^{\prime}\frac{e^{2}|\varphi_{i}(\mathbf{r}^{\prime})|^{2}}{|\mathbf{r}-\mathbf{r}^{\prime}|}\right)\varphi_{i}(\mathbf{r})=E_{i}\varphi_{i}(\mathbf{r})$$

方程左边分别是电子动能，晶格势能以及单电子与其他电子的平均势能。右侧$E_i$是单电子能量也是变分的拉格朗日乘子，由于式子中存在单电子波函数，因此需要给定初始状态，求解出有效势，再代入方程，反复迭代直到收敛（很明显可以使用计算机来处理了）。

由于电子是费米子，满足反对易关系，交换两个电子位置会产生负号，Fork在Hartree的基础上考虑反对易关系，将单电子波函数写成了Slater行列式的形式

$$\psi(\mathbf{q})=\frac{1}{\sqrt{N!}}\begin{vmatrix}\varphi_1(\mathbf{q}_1)&\varphi_1(\mathbf{q}_2)&\cdots&\varphi_1(\mathbf{q}_N)\\\varphi_2(\mathbf{q}_1)&\varphi_2(\mathbf{q}_2)&\cdots&\varphi_2(\mathbf{q}_N)\\\vdots&\vdots&&\vdots\\\varphi_N(\mathbf{q}_1)&\varphi_N(\mathbf{q}_2)&\cdots&\varphi_N(\mathbf{q}_N)\end{vmatrix}$$

此时再将系统总能量变分得到：

$$\begin{gathered}
\left(-\frac{\hbar^{2}}{2m_{e}}\nabla^{2}+V(\mathbf{r})+\sum_{i\neq i^{\prime}}\int\mathrm{d}\mathbf{r}^{\prime}\frac{e^{2}|\varphi_{i}(\mathbf{r}^{\prime})|^{2}}{|\mathbf{r}-\mathbf{r}^{\prime}|}\right)\varphi_{i}(\mathbf{r}) \\
-\sum_{i\neq i^{\prime},||}\int\mathrm{d}\mathbf{r}^{\prime}\frac{e^{2}\varphi_{i}^{*}(\mathbf{r}^{\prime})\varphi_{i}(\mathbf{r}^{\prime})}{|\mathbf{r}-\mathbf{r}^{\prime}|}\varphi_{i}^{\prime}(\mathbf{r})=E_{i}\varphi_{i}(\mathbf{r}) 
\end{gathered}$$

其中左边最后一项表示自旋平行求和。这种方法最重要的地方在于引入了变分和迭代的方法，但还是很难求解，还需要做进一步的近似。

##  Thomas-Fermi-Dirac 近似
Thomas和Fermi发现电子动能可以近似为电子密度的函数，Dirac引入了电子密度泛函的电子交换能修正项，外场和库伦相互作用很明显也是电子密度的函数，动能项和交换项已经被认为可以近似为电子密度的函数，因此电子体系的能量可以被表示为电子密度的泛函（所以说Hartree-Fock引入变分和迭代是非常重要的，提供了很重要的求解思路）。

## Hohenberg-Kohn 定理
Hohenberg-Kohn 提出了两个定理

**定理 1**: 对于不计自旋的全同费米子系统，其基态能量是粒子数密度函数的唯一泛函。

**定理 2**: 在粒子数不变的条件下，能量泛函当且仅当粒子数密度函数 $\rho(r)$ 处于基态时取得极小值，且为基态能量。

根据Thomas-Fermi-Dirac近似和Hohenberg-Kohn定理，每一项都可以表示成电子密度的泛函，基态能量可以被写为

$$\begin{aligned}
E(\rho,V)& =\langle\varphi(\rho)|T+V_{ee}+V_{ext}|\varphi(\rho)\rangle   \\
&=T(\rho)+V_{ee}(\rho)+\int\mathrm{d}\mathbf{r}V_{ext}(\mathbf{r})\rho(\mathbf{r}) \\
&=T(\rho)+\frac{1}{2}\int\mathrm{d}\mathbf{r}\mathrm{d}\mathbf{r}^{\prime}\frac{\rho(\mathbf{r})\rho(\mathbf{r}^{\prime})}{|\mathbf{r}-\mathbf{r}^{\prime}|}+E_{ex}(\rho)+\int\mathrm{d}\mathbf{r}V_{ext}(\mathbf{r})\rho(\mathbf{r})
\end{aligned}$$

其中 $E_{ex}(\rho)$表示电子间的交换关联能。

## Kohn-Sham 方程
但是目前我们还不知道怎么用电子密度表示动能和关联能，Kohn和Sham提出使用无相互作用体系的动能泛函替代真实体系的动能泛函，把二者之差放到关联能中，从而变成单电子问题，$\rho$和$T(\rho)$可以进一步被写为（这里采用了原子单位）：

$$\begin{aligned}\rho(\mathbf{r})&=\sum_i|\varphi_i(\mathbf{r})|^2\\T_0(\rho)&=\sum_i\int\mathrm{d}^3\mathbf{r}\varphi_i^*(\mathbf{r})(-\nabla^2)\varphi_i(\mathbf{r})\end{aligned}$$

这样我们就得到了密度泛函理论的核心方程

$$\left(-\frac{\hbar^{2}}{2m_{e}}\nabla^{2}+V_{ext}(\mathbf{r})+\int\mathrm{d}\mathbf{r}'\frac{e^{2}\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}+\frac{\delta E_{xc}(\rho)}{\delta\rho}\right)\varphi_{i}(\mathbf{r})=E_{i}\varphi_{i}(\mathbf{r})$$

求解这个方程的核心在于找到合适的交换关联泛函，由此可以进一步得到基态能量等

## 参考文献
[1]周丽琴. 拓扑材料及其拓扑相变的第一性原理计算研究[D].中国科学院大学(中国科学院物理研究所),2023.DOI:10.27604/d.cnki.gwlys.2022.000113.
