---
title: "Wigner D-matrix"
collection: notes
permalink: /notes/D_matrix
date: 2024-04-06

---

魏格纳D-矩阵

# Wigner D-matrix

Wigner D-matrix是SU(2)和SO(3)群不可约表示的酉矩阵。由Eugene Wigner在1927年提出，在量子力学的角动量理论中有重要作用。D矩阵的复共轭是球和对称转轴的哈密顿量的本征函数。

## 定义
量子力学中的角动量$J_x,J_y,J_z$是SU(2),SO(3)李代数的生成元，满足对易关系

$$[J_x,J_y]=iJ_z,\quad[J_z,J_x]=iJ_y,\quad[J_y,J_z]=iJ_x,$$

同时

$$J^2=J_x^2+J_y^2+J_z^2$$

与所有生成元对易，可以与$J_z$同时被对角化，所以有基满足

$$J^2|jm\rangle=j(j+1)|jm\rangle,\quad J_z|jm\rangle=m|jm\rangle,$$

其中 j = 0, 1/2, 1, 3/2, 2, ... 对应 SU(2),  j = 0, 1, 2, ... 对应 SO(3). 总的来说, m = −j, −j + 1, ..., j。三维转动算符可以写为欧拉角形式

$$\mathcal{R}(\alpha,\beta,\gamma)=e^{-i\alpha J_z}e^{-i\beta J_y}e^{-i\gamma J_z}$$

Wigner D-matrix是在这个基里的2j+1维矩阵

$$D_{m^{\prime}m}^j(\alpha,\beta,\gamma)\equiv\langle jm^{\prime}|\mathcal{R}(\alpha,\beta,\gamma)|jm\rangle=e^{-im^{\prime}\alpha}d_{m^{\prime}m}^j(\beta)e^{-im\gamma}$$

其中

$$d_{m^{\prime}m}^j(\beta)=\langle jm^{\prime}|e^{-i\beta J_y}|jm\rangle=D_{m^{\prime}m}^j(0,\beta,0)$$

是Wigner (small) d-matrix。

并且$D_{m^{\prime}m}^j(\alpha,0,0)$有与$D_{m^{\prime}m}^j(0,\beta,0)$不同的正交归一性质（对角矩阵）

$$D_{m^{\prime}m}^j(\alpha,0,0)=e^{-im^{\prime}\alpha}\delta_{m^{\prime}m}$$

## 性质

D矩阵的复共轭满足很多性质，可以使用一些算符说明，这些算符是量子力学中空间固定的刚性转子角动量

$$\begin{aligned}
&\hat{\mathcal{J}}_{1} =i\left(\cos\alpha\cot\beta\frac\partial{\partial\alpha}+\sin\alpha\frac\partial{\partial\beta}-\frac{\cos\alpha}{\sin\beta}\frac\partial{\partial\gamma}\right)  \\
&\hat{\mathcal{J}}_{2} =i\left(\sin\alpha\cot\beta\frac{\partial}{\partial\alpha}-\cos\alpha\frac{\partial}{\partial\beta}-\frac{\sin\alpha}{\sin\beta}\frac{\partial}{\partial\gamma}\right)  \\
&\hat{\mathcal{J}}_{3} =-i\frac\partial{\partial\alpha} 
\end{aligned}$$

还有刚体固定的刚性转子角动量

$$\begin{aligned}
&\hat{\mathcal{P}}_{1} =i\left(\frac{\cos\gamma}{\sin\beta}\frac\partial{\partial\alpha}-\sin\gamma\frac\partial{\partial\beta}-\cot\beta\cos\gamma\frac\partial{\partial\gamma}\right)  \\
&\hat{\mathcal{P}}_{2} =i\left(-\frac{\sin\gamma}{\sin\beta}\frac\partial{\partial\alpha}-\cos\gamma\frac\partial{\partial\beta}+\cot\beta\sin\gamma\frac\partial{\partial\gamma}\right)  \\
&\hat{\mathcal{P}_{3}} =-i\frac{\partial}{\partial\gamma}, 
\end{aligned}$$

有对易关系

$$[\mathcal{J}_1,\mathcal{J}_2]=i\mathcal{J}_3,\qquad[\mathcal{P}_1,\mathcal{P}_2]=-i\mathcal{P}_3,$$

还有

$$[\mathcal{P}_i,\mathcal{J}_j]=0,\quad i,j=1,2,3$$

并且平方之和相等

$$\mathcal{J}^2\equiv\mathcal{J}_1^2+\mathcal{J}_2^2+\mathcal{J}_3^2=\mathcal{P}^2\equiv\mathcal{P}_1^2+\mathcal{P}_2^2+\mathcal{P}_3^2$$

具体表达式是

$$\mathcal{J}^2=\mathcal{P}^2=-\frac1{\sin^2\beta}\left(\frac{\partial^2}{\partial\alpha^2}+\frac{\partial^2}{\partial\gamma^2}-2\cos\beta\frac{\partial^2}{\partial\alpha\partial\gamma}\right)-\frac{\partial^2}{\partial\beta^2}-\cot\beta\frac\partial{\partial\beta}$$

$\mathcal{J_i}$作用于D矩阵的第一行

$$\begin{aligned}
\mathcal{J}_{3}D_{m^{\prime}m}^{j}(\alpha,\beta,\gamma)^{*}& =m^{\prime}D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*  \\
(\mathcal{J}_1\pm i\mathcal{J}_2)D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*& =\sqrt{j(j+1)-m^{\prime}(m^{\prime}\pm1)}D_{m^{\prime}\pm1,m}^j(\alpha,\beta,\gamma)^* 
\end{aligned}$$

$\mathcal{P}$作用于矩阵的第二列

$$\mathcal{P}_3D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*=mD_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*$$

注意到由于对易关系不同，因此升降算符也不一样，所以$\mathcal{P}$的被定义为

$$(\mathcal{P}_1\mp i\mathcal{P}_2)D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*=\sqrt{j(j+1)-m(m\pm1)}D_{m^{\prime},m\pm1}^j(\alpha,\beta,\gamma)^*$$

更进一步可以得到

$$\mathcal{J}^2D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*=\mathcal{P}^2D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*=j(j+1)D_{m^{\prime}m}^j(\alpha,\beta,\gamma)^*$$

换句话说，Wigner-D矩阵的复共轭通过$\{\mathcal{J_i}\}$和$\{\mathcal{-P_i}\}$行和列张成的同构李代数的不可约表示

D矩阵还有一个重要性质来源于，$\mathcal{R}$与时间反演算子$T$的作用（其中$T$是anti-unitary的）

$$\langle jm^{\prime}|\mathcal{R}(\alpha,\beta,\gamma)|jm\rangle=\langle jm^{\prime}|T^\dagger\mathcal{R}(\alpha,\beta,\gamma)T|jm\rangle=(-1)^{m^{\prime}-m}\langle j,-m^{\prime}|\mathcal{R}(\alpha,\beta,\gamma)|j,-m\rangle^*$$

也就是

$$D_{m'm}^j(\alpha,\beta,\gamma)=(-1)^{m'-m}D_{-m',-m}^j(\alpha,\beta,\gamma)^*$$

## 正交关系
D矩阵作为欧拉角的函数满足正交关系

$$\begin{aligned}\int_0^{2\pi}d\alpha\int_0^{\pi}d\beta\sin\beta\int_0^{2\pi}d\gamma D_{m^{\prime}k^{\prime}}^{j^{\prime}}(\alpha,\beta,\gamma)^*D_{mk}^j(\alpha,\beta,\gamma)&=\frac{8\pi^2}{2j+1}\delta_{m^{\prime}m}\delta_{k^{\prime}k}\delta_{j^{\prime}j}\end{aligned}$$

这是Schur正交关系的一个特例，根据Peter–Weyl theorem，他们构成一个完全集，$D_{mk}^j(\alpha,\beta,\gamma)$是酉变换（使$\mid lm\rangle $转换到$\mathcal{R}(\alpha,\beta,\gamma)\mid lm\rangle $）的矩阵元，满足

$$\begin{aligned}\sum_kD_{m^{\prime}k}^j(\alpha,\beta,\gamma)^*D_{mk}^j(\alpha,\beta,\gamma)&=\delta_{m,m^{\prime}},\\\sum_kD_{km^{\prime}}^j(\alpha,\beta,\gamma)^*D_{km}^j(\alpha,\beta,\gamma)&=\delta_{m,m^{\prime}}.\end{aligned}$$

SU(2)只依赖于转过的角度$\beta$与转轴无关

$$\chi^j(\beta)\equiv\sum_mD_{mm}^j(\beta)=\sum_md_{mm}^j(\beta)=\frac{\sin\left(\frac{(2j+1)\beta}2\right)}{\sin\left(\frac\beta2\right)}$$

从而满足更简单的正交关系，由群的 Haar measure，可以得到

$$\frac1\pi\int_0^{2\pi}d\beta\sin^2\left(\frac\beta2\right)\chi^j(\beta)\chi^{j^{\prime}}(\beta)=\delta_{j^{\prime}j}$$



## 参考文献
[维基百科](https://en.wikipedia.org/wiki/Wigner_D-matrix)

## Extension
**或许可以用来在某些地方做傅里叶变换的时候当基底使用（？）**