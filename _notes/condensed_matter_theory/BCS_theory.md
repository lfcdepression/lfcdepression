---
title: "BCS Theory of Superconductivity"
collection: notes
permalink: /notes/BCS_theory
date: 2024-03-30

---

关于BCS理论的笔记，主要内容来源于谭志杰老师和熊锐老师固体物理2的ppt和李正中老师的《固体理论》

# 超导的BCS理论
## 一些背景
### 同位素效应
1950年，Maxwell等测量了Hg的几种同位素的$Tc$ ，发现临界转变温度和同位素原子质量有如下关系

$$T_cM^{1/2}=const$$

在其他材料中也存在这种现象，回忆固体物理中关于Debye模型的部分，我们在晶格动力学中有：

$$\omega_D\cdot M^{1/2}=const$$

其中$\omega_D$是Debye频率。两个公式的惊人的相似性让我们可以推导出：当$M\to\infty,T_C\to\infty$，超导态不存在；$M\to\infty,\omega_D\to\infty$，晶格振动不能发生。

很自然地，我们可以猜测：产生超导的原因是电子与晶格振动的相互作用，即电子-声子相互作用

### 电子-声子相互作用的物理图像

电子带负电，两个电子互相排斥，在晶体中，两个电子可通过与晶格振动作用而间接吸引，电子$e_1$ 吸引周围正离子（运动产生声子）向它靠拢,形成正电荷聚集区把附近的另一个电子$e_2$吸引过来。

$e_1$吸引周围正离子$\to$正电荷聚集$\to$吸引$e_2$，相当于两个电子的相互吸引，此过程中，首先晶格离子移动，产生新的晶格振动，而后此振动消失，相当于发射和吸收声子。整个过程可以描述为电子对通过交换声子而产生相互作用。但是这个并不是一个实过程（可以想象一个电子的运动确实很难有那么大的影响，否则Born-Oppenheimer近似就很难发挥作用了），所谓的声子只是电子运动而引起的晶格电子云极化（晶格畸变），因此可以引起另一个极化电子云附近的电子的响应。

## 电子-声子相互作用的哈密顿量

电子-声子相互作用的哈密顿量由三部分组成：单粒子能量，声子-电子相互作用能，电子-电子Coulomb排斥能，这部分的具体推导可以看[电子-声子相互作用](https://lfcdepression.github.io/notes/ep_inter)

$$H=H_0+H_1+u$$

### 单粒子能量项$H_0$

$$H_0=\sum_{\vec{k}\sigma}\varepsilon(\vec{k})C_{\vec{k}\sigma}^{\dagger}C_{\vec{k}\sigma}+\sum_{\vec{q}}\hbar\omega_{\vec{q}}a_{\vec{q}}^{\dagger}a_{\vec{q}}$$

对应于一个无相互作用的系统中，电子和声子的单粒子能量

### 电子和声子相互作用项$H_1$
电子吸收或发射一个声子，从波矢$\vec{k}$ 变为波矢 $\vec{k}+\vec{q}$

$$H_1=\sum_{\vec{q}\vec{k}\sigma}\left\lfloor D_{\vec{q}}C_{\vec{k}+\vec{q},\sigma}^{\dagger}C_{\vec{k}\sigma}a_{\vec{q}}+D_{-\vec{q}}C_{\vec{k}-\vec{q},\sigma}^{\dagger}C_{\vec{k}\sigma}a_{\vec{q}}^+\right\rfloor $$

式中$D_{\vec{q}}=i(\hbar/2M\omega_{\vec{q}})^{1/2}\mid\vec{q}\mid C,M$ 为原子质量，$C$ 为常数，$C\sim-2\varepsilon_F/3$

![1](condensed_matter_theory/BCS_files/image1.png){:height="360px" width="720px"}

左图为波矢为$k$ 的电子吸收一个波矢为$\vec{q}$ 的声子，产生一个波矢为$\vec{k}+\vec{q}$ 的电子，同时，波矢为$\vec{k}$ 的电子，波矢为$\vec{q}$ 的声子被消灭。右图为波矢为$\vec{k}$ 的电子，放出一个波矢为$\vec{q}$ 的声子，变为波矢为$k-\vec{q}$ 的电子。同时波矢为$k$ 的电子被消灭。

### 电子-电子Coulomb排斥作用项$u$

$$u=\sum_{\vec{q}\vec{k}\vec{k}^{\prime}\sigma\sigma^{\prime}}\upsilon_{\vec{q}}C_{\vec{k}-\vec{q},\sigma}^{\dagger}C_{\vec{k}^{\prime}+\vec{q},\sigma^{\prime}}^{\dagger}C_{\vec{k}^{\prime},\sigma^{\prime}}C_{k\sigma}$$

波矢分别为$\vec{k}$ 和$\vec{k}^{\prime}$的一对电子由于Coulomb排斥，交换能量$\hbar\omega_{\vec{q}}$ 后分别处于波矢为$(\vec{k}-\vec{q})$和 $(\vec{k}+\vec{q})$ 的状态。式中$\upsilon_{\vec{q}}$为Coulomb势，应包括屏蔽效应，用一个介电函数描述。



![2](condensed_matter_theory/BCS_files/image2.png){:height="360px" width="720px"}



### 对电子-声子哈密顿量的讨论
- 哈密顿量可以用于各种电子-声子相互作用的讨论
- $H$中包含反对易的电子算符，也包含对易的声子算符，在同一个式子中同时出现对易算符和反对易算符处理有困难
- 系统中声子总数不变（作用通过交换虚声子产生），因此声子自能可以忽略不计，哈密顿量第一项只需要考虑电子自能
- 声子电子相互作用项中，电子对通过交换虚声子间接作用，过程前后各种波矢的声子总数都不变，因此只涉及$H_1$中不包含声子算符的部分$H_{int}$
- $u$中不包含声子算符

因此，我们或许可以写出不包含声子算符的哈密顿量来描述超导体系。具体的操作过程参考[电子-声子相互作用](https://lfcdepression.github.io/notes/ep_inter)，使用中岛变换得到结果

$$H_{int}=\sum_{k,k^{\prime},q,\sigma,\sigma^{\prime}}\frac{\mid D_q\mid^2\hbar \omega_{\vec{q}}}{(\varepsilon_{\vec{k}}-\varepsilon_{\vec{k}+\vec{q}})^2-(\hbar\omega_{\vec{q}})^2}C_{\vec{k}+\vec{q},\sigma}^{\dagger} C_{\vec{k^{\prime}}-\vec{q},\sigma^{\prime}}^{\dagger}C_{\vec{k^{\prime}}\sigma^{\prime}}C_{\vec{k}\sigma}$$

电子-电子间接互作用可能是吸引，也可能是排斥，取决于$(\varepsilon_{\vec{k}}-\varepsilon_{\vec{k}+\vec{q}})^2-(\hbar\omega_{\vec{q}})^2$的符号

### 超导有效哈密顿量
- $H_T=H_0^e+H_{\mathrm{~int~}}+u$不包含声子算符
- 可以证明，只有波矢和自旋都相反的电子对交换声子产生的相互作用最强，因此我们只考虑这种情况

![3](condensed_matter_theory//BCS_files/image3.png){:height="720px" width="320px"}

最后的超导哈密顿量可以写为

$$H_T=H_0^e+\sum_{\vec{k}\vec{k}^{\prime}}V_{\vec{k}\vec{k}^{\prime}}C_{\vec{k}^{\prime}}^{\dagger}C_{-\vec{k}^{\prime}}^{\dagger}C_{-\vec{k}}C_{\vec{k}}$$

式中

$$V_{\vec{k}\vec{k}^{\prime}}=\frac{2\mid D_q\mid^2\hbar \omega_{\vec{q}}}{(\varepsilon_{\vec{k}}-\varepsilon_{\vec{k}+\vec{q}})^2-(\hbar\omega_{\vec{q}})^2}+\upsilon_{\vec{k}-\vec{k'}}$$

前一项来自于间接相互作用，后一项来源于电子间的Coulomb排斥，当$V_{\vec{k}\vec{k}^{\prime}}<0$时为净吸引作用，电子结合成Cooper对，是超导态电子；当$V_{\vec{k}\vec{k}^{\prime}}>0$时，两电子排斥，是正常态电子。

## BCS理论
### Cooper对的能量
电子间有净的吸引作用时，在Fermi面附近动量大小相等，方向相反、自旋相反的两个电子形成的束缚的电子对组态叫Cooper对。Cooper对形成超导电子基态能量肯定低于普通电子。

Cooper讨论了$T=0K$, 在Fermi球外 附加上波矢为$\vec{k_{1}}$和$\vec{k_{2}}$两个电子的情形，费米球内的电子用自由电子气处理，占据球内所有能级，它们与球外的两个电子无相互作用。可对球外两附加电子$\vec{k_1}$和$\vec{k_2}$作二体问题处理。

电子对1, 2自由运动时，波函数用平面波表示，电子对波函数是自由电子波函数 $\varphi_{\vec{k}_1}(\vec{r}_1)$ 和 $\varphi_{\vec{k}_2}(\vec{r}_2)$ 的组合。


$$\phi_{\vec{k}}(\vec{r}_1,\vec{r}_2)=\phi_{\vec{k}_1}(\vec{r}_1)\phi_{\vec{k}_2}(\vec{r}_2)=\frac1{V_c}e^{i(\vec{k}_1\cdot\vec{r}_1+\vec{k}_2\cdot\vec{r}_2)}=\frac1{V_c}e^{i\vec{K}\cdot(\vec{r}_1+\vec{r}_2)/2}e^{i\vec{k}\cdot(\vec{r}_1-\vec{r}_2)}$$

取质心坐标，则有

$$\varphi_{\vec{k}}\left(\vec{r}_{1},\vec{r}_{2}\right)=\frac1{V_{c}}e^{i\vec{k}\cdot(\vec{r}_{1}-\vec{r}_{2})}$$

计入电子1和2的相互作用时，它们就不自由了，不能用一个$\varphi_{\vec{k}}\left(\vec{r}_{1},\vec{r}_{2}\right)$表示，但可展开为所有$\varphi_{\vec{k}}\left(\vec{r}_{1},\vec{r}_{2}\right)$之线性组合。

$$\psi(\vec{r}_1,\vec{r}_2)=\sum_{\vec{k}}g(\vec{k})\varphi_{\vec{k}}(\vec{r}_1,\vec{r}_2)$$

此电子对的Schrödinger方程为 

$$-\frac{\hbar^2}{2m}(\nabla_1^2+\nabla_2^2)\psi(\vec{r}_1,\vec{r}_2)+V(\vec{r}_1,\vec{r}_2)\psi(\vec{r}_1,\vec{r}_2)=(E+2\varepsilon_F)\psi(\vec{r}_1,\vec{r}_2)$$

电子对的能量本征值的意义：Fermi球内能级已有电子占满，外加的两电子只能处于球外的能级，当此两电子在Fermi球面上时，能量为 2$\varepsilon_F$。E 是以 Fermi 面上的电子对的能量作零点算得的电子对能量。

取相对坐标，代入薛定谔方程可以得到

$$-\frac{\hbar^2}m\nabla^2\psi(\vec{r})+V(\vec{r})\psi(\vec{r})=(E+2\varepsilon_F)\psi(\vec{r})$$

将线性组合的式子代入薛定谔方程，并且两边同时乘上$\psi^*$对全空间积分可以得到：

$$\frac{\hbar^2k^2}mg(\vec{k})+\sum_{\vec{k}^{\prime}}g(\vec{k}^{\prime})V_{\vec{k},\vec{k}^{\prime}}=(E+2\varepsilon_F)g(\vec{k})$$

其中

$$V_{\vec{k},\vec{k}^{\prime}}=\frac1{V_{c}}\int V(\vec{r})e^{i(\vec{k}-\vec{k^{\prime}})\vec{r}}d^3r$$

上式是 $\vec{k}$ 空间的Schrödinger方程，$g(\vec{k})$相当于 $\vec{k}$空间的波函数，解此方程关键是确定矩阵元$V_{\vec{k},\vec{k}^{\prime}}$，所以我们需要简化这个式子，$\vec{k}$和$\vec{k^{\prime}}$是单电子相对波矢，两个电子都处于Fermi球外，故电子通过交换虚声子$\vec{q}$改变波矢，而$q\leq q_{D}$，（$q_{D}$是Debye 波矢，是声子的最大波矢）,故 $k_F<\mid \vec{k}\mid ,\mid \vec{k}^{\prime}\mid<k_F+q_D$如取$\varepsilon_F=0$,则为 $0<\varepsilon(\vec{k}),\varepsilon(\vec{k^{\prime}})<\hbar\omega_D$。又有$k_{F}>q_{D}$,故$\vec{k},\vec{k}^{\prime}$ 只在很小范围内变化。

因此可以取近似简化为

$$V_{\vec{k},\vec{k}^{\prime}}=\begin{cases}-V&\quad k_F<|\vec{k}\mid,|\vec{k^{\prime}}|<k_F+q_D\\0&\quad\text{其他情况}&\end{cases}$$

因此我们得到

$$\left[E+2\varepsilon_F-\frac{\hbar^2k^2}m\right]g(\vec{k})=-V\sum_{\vec{k}^{\prime}}g(\vec{k}^{\prime})=-VC$$

波函数可以写为

$$g(\vec{k})=\frac{CV}{2\frac{\hbar^2k^2}{2m}-2\varepsilon_F-E}=\frac{CV}{2\varepsilon(\vec{k})-E}$$

其中$\varepsilon(k)=\frac{\hbar^2k^2}{2m}-\varepsilon_F$是以$\varepsilon_F$为零点的一个自由电子的能量。

再求和可以得到：

$$\begin{aligned}CV\sum_{\vec{k}}\frac{1}{2\varepsilon(\vec{k})-E}&=\sum_{\vec{k}}g(\vec{k})=C\\\\V\sum_{\vec{k}}\frac{1}{2\varepsilon(\vec{k})-E}&=1\end{aligned}$$

$E$处在求和号下面，要求出电子对的能量$E$ 需先对$\vec{k}$ 求和由于$\vec{k}$与 $\varepsilon(k)$一一对应，可用对能量空间的状态积分代替

$$\sum_k\to\int N(\varepsilon)d\varepsilon $$

根据前面的条件我们可以得到

$$V\int_0^{\hbar\omega_D}\frac1{2\varepsilon-E}N\left(\varepsilon\right)d\varepsilon=1$$

在数量级上$\varepsilon_F\sim1-10eV,\hbar\omega_D\sim10^{-2}eV$,有 $h\omega_D<<\varepsilon_F$ 即能量$\varepsilon$ 的变化范围很窄，可用Fermi面附近状态密度 $N(0)$代替 $N(\varepsilon)$

进行这个代换之后积分可以得到

$$\frac{N(0)V}2\mathrm{ln}\frac{E-2\hbar\omega_D}E=1$$

也就是

$$E=2\hbar\omega_D{\left[1-\exp\left(\frac2{N(0)V}\right)\right]^{-1}}$$

弱耦合情况下我们有

$$E=-2\hbar\omega_D\exp\left(-\frac2{N(0)V}\right){<0}$$

### 什么电子形成Cooper对

首先必须满足动量守恒和能量守恒

$$\hbar(\vec{k_1}+\vec{k_2})=\hbar(\vec{k_1^{\prime}}+\vec{k_2^{\prime}})=\hbar\vec{K}$$

由于交换虚声子产生作用，因此能量受到限制

$$\left(\hbar\omega_q\right)_{\mathrm{max}}=\hbar\omega_D$$

结合这两个条件我们有$\Delta k$需要满足$\varepsilon=\frac{\hbar^2k^2}{2m}$得到$\Delta\varepsilon=\frac{1}{m}\hbar^2k\Delta k$，也就是$\Delta k=\frac{m}{\hbar^2k}\Delta\varepsilon$，而极限情况下$\Delta\varepsilon=\hbar\omega_D$，所以最大的$\Delta k=\frac{m\omega_D}{\hbar k_F}$

因此$:\vec{k}_1,\vec{k}_2,\vec{k}_1^{\prime},\vec{k}_2^{\prime}$只有落在两球壳交叠区才能同时满足 


![4](condensed_matter_theory/BCS_files/image4.png){:height="360px" width="720px"}

### Bardeen, Cooper, Schrieffer Theory























