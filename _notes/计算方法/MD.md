---
title: "A Preliminary of Molecular Dynamics"
collection: notes
permalink: /notes/MD
date: 2024-03-09

---
为了学习机器学习力场相关内容，自学分子动力学的一个笔记。

主要参考李新征老师《Computer Simulations of Molecules and Condensed Matters:From Electronic Structure to Molecular Dynamics》

# 分子动力学
## 哈密顿力学
### 哈密顿正则方程
基本思路：经典力学由牛顿第二定律主导：\\(m_i\dot{\mathbf{r}}_i=\mathbf{F}_i\\)，这是一个矢量二阶微分方程，若在d维空间中有N个粒子，那么我们一共有dN个方程，通过选取初始的r和v可以得到唯一的解。经典力学产生了一类经典的动力系统：哈密顿系统，研究在相空间中解的性态，相空间中的演化由哈密顿正则方程主导：

$$\begin{aligned}\dot{p}_i&=-\frac{\partial H}{\partial q_i}\\\dot{q}_i&=\frac{\partial H}{\partial p_i}\end{aligned}$$

### 刘维尔定理
刘维尔定理是哈密顿动力系统中的重要定理，其内容是：保守力学体系在相空间中的代表点的密度在运动中保持不变。对于无耗散系统我们有时间反演对称性以及哈密顿量对时间的导数为0

$$\begin{aligned}\frac{\mathrm{d}H}{\mathrm{d}t}&=\sum_{i=1}^N\left[\frac{\partial H}{\partial\mathbf{r}_i}\mathbf{\dot{r}_i}+\frac{\partial H}{\partial\mathbf{p}_i}\mathbf{\dot{p}_i}\right]\\&=\sum_{i=1}^N\left[\frac{\partial H}{\partial\mathbf{r}_i}\frac{\partial H}{\partial\mathbf{p}_i}-\frac{\partial H}{\partial\mathbf{p}_i}\frac{\partial H}{\partial\mathbf{r}_i}\right]=0\end{aligned}$$


根据连续性方程可以得到：

$$\frac{\partial\rho}{\partial t}+\nabla\cdot\boldsymbol{j}=0$$

类比电流可以得到相空间的“流密度”

$$\nabla\cdot\boldsymbol{j}=\sum_\alpha\left(\frac{\partial(\rho\boldsymbol{v})_{q_\alpha}}{\partial q_\alpha}+\frac{\partial(\rho\boldsymbol{v})_{p_\alpha}}{\partial p_\alpha}\right)$$

进一步展开可以得到

$$\nabla\cdot\boldsymbol{j}=\sum_\alpha\left(\frac{\partial\rho}{\partial q_\alpha}\dot{q_\alpha}+\rho\frac{\partial\dot{q_\alpha}}{\partial q_\alpha}+\frac{\partial\rho}{\partial p_\alpha}\dot{p_\alpha}+\rho\frac{\partial\dot{p_\alpha}}{\partial p_\alpha}\right)$$

根据哈密顿正则方程的限制我们可以进一步简化上面的公式得到


$$\nabla\cdot\boldsymbol{j}=\sum_\alpha\left(\frac{\partial\rho}{\partial q_\alpha}\dot{q_\alpha}+\frac{\partial\rho}{\partial p_\alpha}\dot{p_\alpha}\right)$$

结合连续性方程，这个式子可以表示为

$$\frac{\partial\rho}{\partial t}+\sum_\alpha\left(\frac{\partial\rho}{\partial q_\alpha}\dot{q_\alpha}+\frac{\partial\rho}{\partial p_\alpha}\dot{p_\alpha}\right)=0$$

## Verlet 算法
### 传统Verlet算法
数值模拟需要离散化来处理微分方程，常用的方法是使用泰勒展开来做：

$$\begin{aligned}
&r(t+\Delta t)\\&=r(t)+\nu(t)\Delta t+\frac{F(r(t))}{2m}\Delta t^{2}+\frac{1}{6}\left.\frac{\partial^{3}\boldsymbol{r}}{\partial t^{3}}\right|_{t}\Delta t^{3}+O(\Delta t^{4}) \\ \\
&\boldsymbol{r}(t-\Delta t)\\&=\boldsymbol{r}(t)-\boldsymbol{\nu}(t)\Delta t+\frac{\boldsymbol{F}(\boldsymbol{r}(t))}{2\boldsymbol{m}}\Delta t^{2}\left.-\frac{1}{6}\frac{\partial^{3}\boldsymbol{r}}{\partial t^{3}}\right|_{t}\Delta t^{3}+O(\Delta t^{4})
\end{aligned}$$

通过上面两个式子相加，相减可以得到下一个时刻的坐标\\(\boldsymbol{r}(t+\Delta t)\\)和这一时刻的速度\\(\boldsymbol{v}(t)\\)，得到下面一个时刻的坐标需要之前的两个时刻的坐标，但是我们一般只有一个初始条件，因此这里有一个技巧先得到低精度再得到高精度

$$\begin{aligned}\boldsymbol{r}(t+\Delta t)&=2\boldsymbol{r}(t)-\boldsymbol{r}(t-\Delta t)+\frac{\boldsymbol{F}(\boldsymbol{r}(t))}{\boldsymbol{m}}\Delta t^2+O(\Delta t^4)\\ \\ \boldsymbol{v}(t)&=\frac{'\boldsymbol{r}(t+\Delta t)-\boldsymbol{r}(t-\Delta t)}{2\Delta t}+O(\Delta t^2)\end{aligned}$$

### 速度Verlet算法
把力替换成势能的梯度，之后第一步匀加速直线运动更新半个时间小量的速度，使用更新的速度做匀速直线运动更新坐标，之后再更新最后的速度，这样保证坐标和速度精度和Verlet一样，同时我们绕开了Verlet中初值条件问题和速度更新问题

$$\begin{aligned}
&\boldsymbol{v}(t+\Delta t/2)=\boldsymbol{v}(t)-\left(\frac{\partial V}{\partial\boldsymbol{r}(t)}\right)\frac{\Delta t}{2m}+O(\Delta t^{3}) \\
&\boldsymbol{r}(t+\Delta t)=\boldsymbol{r}(t)+\boldsymbol{v}(t+\Delta t/2)\Delta t+O(\Delta t^{4}); \\
&{\boldsymbol{v}}(t+\Delta t)={\boldsymbol{v}}(t+\Delta t/2)-\left({\frac{\partial V}{\partial{\boldsymbol{r}}(t+\Delta t)}}\right){\frac{\Delta t}{2m}}+{\boldsymbol{O}}(\Delta t^{2})
\end{aligned}$$

### 为什么Verlet算法是有效的呢？
密度可以写成\\( \rho =\frac{\exp(\frac{-H}{kT})}{Z}\\)是不显含时间的，之后把泊松括号作为一个算符来处理（算符化之后有很多经常用在量子力学里面的方法可以被我们利用处理问题，例如Trotter展开等）

$$\begin{aligned}iL&=\sum_{i=1}^{N}\left[\frac{\partial H}{\partial\mathbf{r}_i}\cdot\frac{\partial}{\partial\mathbf{r}_i}-\frac{\partial H}{\partial\mathbf{r}_i}\cdot\frac{\partial}{\partial\mathbf{p}_i}\right]\\&=\sum_{i=1}^{N}\left[\frac{\mathbf{p}_i}{m_i}\cdot\frac{\partial}{\partial\mathbf{r}_i}+\mathbf{F}_i\cdot\frac{\partial}{\partial\mathbf{p}_i}\right]\end{aligned}$$


Trotter展开是保辛的，在哈密顿系统中保辛和雅可比行列式为0是等价因此，使用Trotter展开误差是有界的，只要步长不太大，那么就可以保证长时间的演化是没有太大误差的。

$$J(\mathbf{x}_{\Delta t},\mathbf{x}_0)=\frac{\partial(x_{\Delta t}^1,\ldots,x_{\Delta t}^n)}{\partial(x_0^1,\ldots,x_0^n)}$$

下面我们开始使用这套方法处理我们研究的系统，首先使用Trotter展开

$$\boldsymbol{x}(t)=e^{i\hat{L}\{x\}t}\boldsymbol{x}(0)\approx e^{i\hat{L}_2\{x\}t/2}e^{i\hat{L}_1\{x\}t}e^{i\hat{L}_2\{x\}t/2}\boldsymbol{x}(0)$$

再取线性项

$$\begin{aligned}e^{i\hat{L}_2(\mathbf{x})\Delta t/2}\boldsymbol{x}(t_0)&=\left[1+i\hat{L}_2\frac{\Delta t}2+\frac1{2!}(i\hat{L}_2)^2(\frac{\Delta t}2)^2+\cdots\right]\boldsymbol{x}(t_0)\\&\approx\left(1+i\hat{L}_2\frac{\Delta t}2\right)\boldsymbol{x}(t_0)\end{aligned}$$

最终得到

$$e^{i\hat{L}_2\{\boldsymbol{x}\}\Delta t/2}\boldsymbol{x}(t_0)=\boldsymbol{x}(t_0)+\frac{\Delta t}2\sum_{i=1}^{3N}\dot{p}_i\frac{\partial\boldsymbol{x}(t_0)}{\partial\boldsymbol{p}_i}$$

结合过程还可以得到\\(\hat{L}_2\\)的物理意义

$$i\hat{L}_2=-\sum_{i=1}^{3N}\frac{\partial H}{\partial x_i}\frac{\partial}{\partial p_i}$$

$$e^{i\hat{L}_2\Delta t/2}\boldsymbol{x}_0=\left\{x_1^0,x_2^0,\cdots,x_{3N}^0,\left\{p_i+\frac{\Delta t}2\dot{p}_i\right\}_{i=1}^{3N}\right\}$$

代入动量的导数与势能梯度的关系，我们最终可以得到

$$e^{i\hat{L}_2\Delta t/2}\boldsymbol{x}_0=x_0(\boldsymbol{q}(t_0);\boldsymbol{p}(t_0+\frac{\Delta t}2))$$

很神奇的：这是速度Verlet算法的第一步，我们发现了Verlet算法有效的原因，只要波动不太大，时间步长足够小，使用Trotter展开和Taylor展开的方法就是比较准确的。同理我们可以得到：

$$\begin{aligned}&e^{i\hat{L}_1\Delta t}\boldsymbol{x}(t_0+\frac{\Delta t}2)\\&=\left\{\left\{x_i+\Delta t\dot{x}_i\right\}_{i=1}^{3N};p_1^{t_0+\frac{\Delta t}2},p_2^{t_0+\frac{\Delta t}2},\cdots,p_{3 N}^{t_0+\frac{\Delta t}2}\right\}\\&=\left\{\left\{x_i+\frac{p_i}m\Delta t\right\}_{i=1}^{3N};p_1^{t_0+\frac{\Delta t}2},p_2^{t_0+\frac{\Delta t}2},\cdots,p_{3N}^{t_0+\frac{\Delta t}2}\right\}\\&=x(\boldsymbol{q}(t_0+\Delta t);\boldsymbol{p}(t_0+\frac{\Delta t}2))\end{aligned}$$


这是速度Verlet算法的第二步，同理可以得到第三步演化的公式

$$\begin{aligned}&e^{i\hat{L}_2\Delta t/2}\boldsymbol{x}(\boldsymbol{q}(t_0+\Delta t);\boldsymbol{p}(t_0+\frac{\Delta t}2))\\&=\boldsymbol{x}(\boldsymbol{q}(t_0+\Delta t);\boldsymbol{p}(t_0+\Delta t))\end{aligned}$$


到这里我们可以看出，Verlet算法是在精确演化的基础上，使用Trotter展开之后，对e指数进行Taylor展开并取线性项的结果。


## 分子动力学与机器学习力场
### 基本思路
学习化学结构与势能之间的统计关系，而不依赖于先入为主的固定化学键概念或相关相互作用的知识，来达到结合ab initio方法的准确性和传统力场方法的效率的目的，加速分子动力学模拟过程。

大多数传统的立场将势能写为bonded和nonbonded项，第一项可以使用一个简单的与距离、角度等的函数，后一项经常用库仑定律来描述，有一种比较常用的是Lennard-Jones势。当极化或多体效应比较显著时，上面的方法并不能很好的描述这个过程，虽然也有一些办法可以解决，但这些不总是先验（priori）的。并且传统的力场模型无法描述成键或者断键，也有一些方法可以解决，但是效果和普适性都比较差。还有一种和量子力学结合的方法，这种方法的问题是计算成本太高。

构建ML-FFs时，需要大量优质数据来了解相关的结构-功能关系，这些数据需要ab initio的计算得来，不同于传统的方法，这种方法基本不需要物理，几乎是数据驱动的，因此可以绕开很多假设和限制，直接从原子的排列结构得到力场。
### 物理化学基础

最准确的方法是使用薛定谔方程来描述，但是我们能解的只有氢原子系统，其他的都需要数值解，甚至数值解都很难做，所以发展出了很多的近似（比如波恩奥本海默近似，我们不需要考虑原子核的运动，只需要考虑电子就行，电子处在一个由原子核等产生的势场当中）
#### 势能面
BO近似暗示了存在一个核电荷数Z和距离r对于能量E的一个函数关系，这个函数叫做势能面（potential energy surface(PES)）,势能面的谷是稳定结构，从一个谷到另外一个谷往往伴随着化学反应或者结构变化。有了PES就可以求势能梯度，也就可以进行分子动力学模拟了。

虽然BO近似简化了我们需要求解的薛定谔方程，但是仍然是无法计算的，因此多数情况下在一个分子动力学模拟中在每一步演化计算ab initio的能量和力是做不到的。所以经常使用一个解析函数来表示PES，于是问题转向了如何找到合适的方程和参数
### 机器学习的一些要求
- 能量守恒：ML-FFs需要确保在没有外部力作用的情况下，化学系统的总能量是守恒的。这意味着预测的力场必须是保守的，即力是势能的负梯度。
- 旋转和平移不变性：ML-FFs应保持旋转和平移不变性，这意味着势能仅依赖于原子之间的相对位置，而与它们的绝对位置无关。
- 原子不可区分性：在Born-Oppenheimer近似下，具有相同核电荷的原子是不可区分的。ML-FFs应确保对原子的置换具有对称性，以反映这种物理不变性。
- 势能面（PES）：ML-FFs的目标是学习原子位置与势能之间的映射关系。这允许ML模型预测分子的稳定性、反应路径以及热力学和动力学性质。
- 核量子效应（NQEs）：在处理轻原子或强键系统时，核量子效应变得重要。ML-FFs可以被训练来包含这些效应，从而提供更准确的模拟结果。
- 电子效应：ML-FFs能够捕捉到电子效应对分子动力学的影响，这对于理解和预测分子行为至关重要。
## 参考文献
1. [李永乐. 《计算物理学》](https://www.koushare.com/video/videodetail/24883)
2. Li, Xin-Zheng, and En-Ge Wang. Computer Simulations of Molecules and Condensed Matter: From Electronic Structures to Molecular Dynamics. 2018.
3. Unke, Oliver T., et al. "Machine learning force fields." Chemical Reviews 121.16 (2021): 10142-10186.
