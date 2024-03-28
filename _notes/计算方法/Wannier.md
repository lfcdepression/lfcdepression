---
title: "A Preliminary of Wannier Function"
collection: notes
permalink: /notes/Wannier
date: 2024-03-25

---
一个关于Wannier函数方法的笔记

# Wannier函数

根据布洛赫定理在周期性势场中我们有

$$\begin{aligned}\varphi_n(\mathbf{k},\mathbf{r})&=e^{i\mathbf{k}\cdot\mathbf{r}}u_n(\mathbf{k},\mathbf{r})\\u_n(\mathbf{k},\mathbf{r})&=u_n(\mathbf{k},\mathbf{r}+\mathbf{R}_n)\end{aligned}$$

对布洛赫波函数做傅立叶变换就可以得到实空间的万尼尔函数：

$$\begin{gathered}
\omega_{n\mathbf{R}}(\mathbf{r})=\frac{V}{(2\pi)^{3}}\int_{\mathrm{BZ}}\mathrm{dk}e^{-i\mathbf{k}\cdot\mathbf{R}}\varphi_{n\mathbf{k}}(\mathbf{r}) \\
\varphi_{n\mathbf{k}}(\mathbf{r})=\sum_{\mathbf{R}}e^{-i\mathbf{k}\cdot\mathbf{R}}\omega_{n\mathbf{R}}(\mathbf{r}) 
\end{gathered}$$

在处理一些材料的表面性质或者强关联体系的问题时使用万尼尔函数会更加方便一些。在多能带系统中变换一般形式为：

$$u_{n\mathbf{k}}(\mathbf{r})\to\sum_{m}U_{mn}(\mathbf{k})u_{m\mathbf{k}}(\mathbf{r})$$

此时布洛赫波函数的傅立叶变换为：

$$\omega_{n\mathbf{R}}(\mathbf{r})=\frac{V}{(2\pi)^3}\int_{\mathrm{BZ}}\mathrm{d}\mathbf{k}e^{-i\mathbf{k}\cdot\mathbf{R}}\left[\sum_{m}U_{mn}(\mathbf{k})\varphi_{m\mathbf{k}}(\mathbf{r})\right]$$

由于\\(U_{mn}(\mathbf{k})\\)的存在导致Wannier函数具有不确定性，可以通过求解最局域的Wannier函数解决。

## 最大局域化万尼尔函数

要想得到最局域的万尼尔函数，就需要将\\(U_{mn}(\mathbf{k})\\)最小化，为此可以使用SMV算法，在实空间定义一个表示Wannier函数展宽的函数\\(\Omega\\)，通过求其极小值来求最局域Wannier函数：

$$\begin{aligned}
\Omega& =\sum_{n}\left[\langle\mathbf{r}^{2}\rangle_{n}-\langle\mathbf{r}\rangle_{n}^{2}\right]  \\
&=\sum_{n}\left(\langle\omega_{n0}(\mathbf{r})|\mathbf{r}^{2}|\omega_{n0}(\mathbf{r})\rangle-|\langle\omega_{n0}(\mathbf{r})|\mathbf{r}|\omega_{n0}(\mathbf{r})\rangle|^{2}\right)
\end{aligned}$$

\\(\Omega\\)可以分为规范不变部分和以来相位的规范变换项，规范变换项又可以被分为对角项和非对角项，可以表示为

$$\begin{aligned}\Omega&=\Omega_I+\tilde{\Omega}\\&=\Omega_I+\Omega_D+\Omega_{OD}\end{aligned}$$

三项分别为：

$$\begin{gathered}
\Omega_{I}=\sum_{n}\left[\langle\omega_{n0}(\mathbf{r})|\mathbf{r}^{2}|\omega_{n0}(\mathbf{r})\rangle-\sum_{m\mathbf{R}}|\langle\omega_{m\mathbf{R}}(\mathbf{r})|\mathbf{r}|\omega_{n0}(\mathbf{r})\rangle|^{2}\right] \\
\Omega_{D}=\sum_{n}\sum_{\mathbf{R}\neq0}|\langle\omega_{n\mathbf{R}}(\mathbf{r})|\mathbf{r}|\omega_{n0}(\mathbf{r})\rangle|^{2} \\
\Omega_{OD}=\sum_{m\neq n}\sum_{\mathbf{R}}|\langle\omega_{m\mathbf{R}}(\mathbf{r})|\mathbf{r}|\omega_{n0}(\mathbf{r})\rangle|^{2} 
\end{gathered}$$

有了实空间的表述，我们可以在k空间中也得到相应的表达式：

$$\begin{gathered}
\langle\mathbf{r}\rangle_{n}={\frac{i}{N}}\sum_{\mathbf{k},\mathbf{b}}\omega_{\mathbf{b}}\mathbf{b}\left[\langle u_{n\mathbf{k}}|u_{n,\mathbf{k}+\mathbf{b}}\rangle-1\right] \\
\langle\mathbf{r}^{2}\rangle_{n}={\frac{1}{N}}\sum_{\mathbf{k},\mathbf{b}}\omega_{\mathbf{b}}\left[2-2\mathrm{Re}\langle u_{n\mathbf{k}}|u_{n,\mathbf{k}+\mathbf{b}}\rangle\right] 
\end{gathered}$$

\\(b\\)是链接每个k和k'的矢量，\\(\omega\\)是权重，可以在离散化的布里渊区中定义交叠矩阵

$$M_{mn}^{(\mathbf{k},\mathbf{b})}=\langle u_{m\mathbf{k}}|u_{n,\mathbf{k}+\mathbf{b}}\rangle$$ 

之后可以得到

$$\begin{gathered}
\Omega_{I} =\frac{1}{N}\sum_{\mathbf{k},\mathbf{b}}\omega_{b}\left(J-\sum_{mn}|M_{mn}^{(\mathbf{k},\mathbf{b})}|^{2}\right)=\frac{1}{N}\sum_{\mathbf{k},\mathbf{b}}\omega_{b}\mathrm{tr}\left[P^{(\mathbf{k})}Q^{(\mathbf{k}+\mathbf{b})}\right] \\
\Omega_{D} =\frac{1}{N}\sum_{\mathbf{k},\mathbf{b}}\omega_{b}\sum_{n}\left(-\mathrm{Im}(\ln M_{nn}^{(\mathbf{k},\mathbf{b})})-\mathbf{b}\cdot\bar{\mathbf{r}}_{n}\right)^{2} \\
\Omega_{OD}=\frac{1}{N}\sum_{\mathbf{k},\mathbf{b}}\omega_{b}\sum_{m\neq n}|M_{mn}^{(\mathbf{k},\mathbf{b})}|^{2} 
\end{gathered}$$

其中k 点处布洛赫子空间的规范不变的投影算符为

 $$P^{(\mathbf{k})}=\sum_n|u_{n\mathbf{k}}\rangle\langle u_{n\mathbf{k}}|$$ 
 
\\(Q^{(\mathbf{k})}=1-P^{(\mathbf{k})}\\),能带指标 \\(m,n=1,2,...,J\\)。先看 \\(\Omega_I\\) 项，如果 k 点和 k'点离得越近，其波函数的交叠矩阵的平方越大，\\(\Omega_I\\) 就越小；当它们完全重叠时，该项为 0, 因此\\(Tr([P^{(\mathrm{k})}Q^{(\mathrm{k}+\mathrm{b})}])\\) 测量的是 k 和 k+ b 的相邻布洛赫子空间之间的不匹配 (或“溢出”) 程度，这样我们就只需要优化 \\(\widetilde{\Omega}\\) 来得到最局域的万尼尔函数。

最小化 \\(\Omega=\Omega_I+\widetilde{\Omega}\\) 的过程就是同时对希尔伯特子空间的选择和对规范选择的优化。优化完成后我们可以得到一组幺正矩阵\\(U(\mathbf{k})\\)满足：


$$u_{n\mathbf{k}}^W=\sum_{m=1}^NU_{mn}(\mathbf{k})u_{m\mathbf{k}}$$

做傅里叶变换即可得到实空间最大局域的Wannier 函数，如果对布洛赫表象下的哈密顿量做类似的幺正变换，再作傅里叶变换块就可以得到实空间中的紧束缚哈密顿量 


$$\begin{aligned}H_{\mathbf{k}}&=U^\dagger(\mathbf{k})H(\mathbf{k})U(\mathbf{k})\\\\H_{mn}^W(\mathbf{R})&=\frac{1}{N}\sum_{\mathbf{k}}e^{-i\mathbf{k}\cdot\mathbf{R}}H_{mn}^W(\mathbf{k})\equiv\langle m0|\hat{H}|n\mathbf{R}\rangle\end{aligned}$$
