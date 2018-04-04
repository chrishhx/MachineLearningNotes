# 支持向量机 Support Vector Machine

## 概述
支持向量机(SVM)是Corinna Cortes和Vapnik等人提出的，它在解决小样本、非线性及高维模式识别中表现出许多特有的优势，并能够推广应用到函数拟合等其他机器学习问题中。在机器学习中，支持向量机是与相关的学习算法有关的监督学习模型，常用作分类器。

## 最大间隔分类器 Maximal Margin Classifier
p维欧氏空间 $\mathbb{R}^p$ 上的超平面可以表示为
$$
\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p = 0
$$

超平面可以简记为$ X^T \beta + \beta_0 = 0 $，其中$\beta = (\beta_1,...,\beta_p)$，对 $\mathbb{R}^p$上的一个点 $x = (x_1,...,x_p)^T$
$$
\left \lbrace
\begin{array}{lll}
x^T \beta + \beta_0 < 0\ ,\ if\ x\ is\ under\ the\ hyperplane\\ \\
x^T \beta + \beta_0 = 0\ ,\ if\ x\ is\ on\ the\ hyperplane \\ \\
x^T \beta + \beta_0 > 0\ ,\ if\ x\ is\ above\ the\ hyperplane
\end{array} \right .
$$ 由此得到了$\mathbb{R}^p$的一个划分。

假设有n个p维的训练样本观测
$$ x_1=(x_{1,1},...,x_{1,p})^T , ...... , x_n=(x_{n,1},...x_{n,p})^T $$
这些观测可以分为两类，类标签记作 $y_1,...,y_n \in \{-1,1\}$，将训练样本观测表示为$n*p$的矩阵则有
$$
X = \begin{bmatrix}
x_{1}^T \\ \\
\vdots \\ \\
x_{n}^T
\end{bmatrix} = 
\begin{bmatrix}
x_{1,1} & \cdots & x_{1,p} \\ \\
\vdots & \ddots & \vdots \\ \\
x_{n,p} & \cdots & x_{n,p}
\end{bmatrix} \ \ \ \ 
Y = \begin{bmatrix}
y_{1} \\ \\
\vdots \\ \\
y_{n}
\end{bmatrix} 
$$

分类的目标是将测试样本观测$x^*=(X_1^*,...,X_p^*)$正确的分类，为了实现这个分类，引入基于分离平面的方法：
假设可以找到一个超平面 $H=\{X|\beta_0+\beta X=0\}$ ，使得训练样本可以准确无误地分成符合标签两类
$$
\left \lbrace
\begin{array}{lll}
x_i^T \beta + \beta_0 < 0 \ \ , if\ \ y_i=-1 \\ \\
x_i^T \beta + \beta_0 > 0 \ \ , if\ \ y_i=1
\end{array} \right.
\ \ \forall i \in \{1,...,n\}
$$ 等价地，可以把这两个式子综合为：

$$
y_i(x_i^T \beta + \beta_0) > 0\ \ \ \forall i \in \{1,...,n\} 
$$

这就是分离平面 $x_i^T \beta + \beta_0 = 0$ 将所有训练样本正确地划分到两侧的充要条件。


如果这样的分离平面存在，就可以很自然地用它建立一个二分类器，分类的结果取决于测试样本点落在超平面的哪一测，即根据 $f(x) = x_i^T \beta + \beta_0$的符号分类，如果$f(x^*)$为负，则将$x$分类到 -1 类，如果$f(x)$为正，则将$x$分类到 1 类。另外，$|f(x^*)|$ 的大小也有一定的意义，如果$|f(x^*)| \gg 0$，说明这个点远离分离平面，意味着这个分类结果置信度更高。反过来说，如果 $|f(x^*)|$ 比较小，那么这个分类结果的置信度就比较低。

一般来说，如果样本是线性可分的，那么分离超平面很可能不止一个，即对于某一个分离超平面$H$，稍微调整之后得到的超平面 $H' = H + \epsilon$ 依然可以将样本准确划分。为了构建一个基于分离超平面的分类器，我们需要一个合适的标准。

我们可以计算所有观测样本到超平面的距离 $d_i$ ，找出其中距离最小的一个样本，将它到超平面的距离称为边距（margin） $M$。一个很自然的分离超平面的选择就是使得边距最大的超平面，称为最大边距超平面（maximal margin hyperplane），也称为最佳分离超平面（optimal separating hyperplane）。如果将这个的最大边距超平面用于分类得到的分类器，称为最大边距分类器（maximal margin classifier），又称硬边界的支持向量机（hard margin support vector machine）。需注意的是在特征维数 $p$ 足够大的时候，比如 $p = n$ 时，这个分类器很有可能会过拟合，因此要获得合适的分离平面，要考虑添加适当的正则化项以调整训练目标。

给定训练样本 $ x_1,...,x_n \in \mathbb{R}^p$ ，和分类标签 $y_1,...,y_n \in \{-1,1\}$，那么最大边距分类器就是如下问题的解：
$$
\underset{\beta_0,...,\beta_p}{\mathrm{maximize}} M
$$

$$
\mathrm{subject\ to}\ \ \frac{y_i(x_i^T \beta + \beta_0)}{||\beta||^2} \geq M > 0 \ \ \ \ \forall i\in\{1,...,n\}
$$

由于$(\beta_1,...,\beta_p)^T$ 为超平面的法向量，则样本 $x_i$ 到超平面的距离为
$$
m_i = \frac{|x_i^T \beta + \beta_0|}{||\beta||^2} = \frac{y_i(x_i^T \beta + \beta_0)}{||\beta||^2}
$$

限制 $m_i > M$ 就是要求所有的样本都被正确地分到分离超平面的两侧，结合前面即有
$$ M = \mathrm{min}\ m_i\ ,\ i\ \in \{1,...,n\}$$

最大化边距的问题也就变成了最小化$||\beta||^2$，优化问题可以改写为：
$$
\begin{array}{ccc}
\underset{\beta,\beta_0}{min}\ ||\beta||^2\\ \\
\mathrm{subject\ to}\ y_i(x_i^T \beta + \beta_0) > 0 ,\ \forall i=1,...,n
\end{array}
$$

从优化问题的角度看，优化的目标函数是二次项，那么可以添加一个常数系数方便求导时约去，开区间限制没那么好处理，也可以稍微改一下，最终可以将问题修正为
$$
\begin{array}{ccc}
\underset{\beta,\beta_0}{min}\ \dfrac{1}{2}||\beta||^2\\ \\
\mathrm{subject\ to}\ y_i(x_i^T \beta + \beta_0) \geq 1 ,\ \forall i=1,...,n
\end{array}
$$

第一项乘一个常数系数不影响解，而第二项相当于改变了求解的空间，要求分离平面距离两个分类的点都必须有一段距离，即样本点不能落在分离平面上，而这其实对求解空间的影响也非常小，因为前面就假设了样本必须是线性可分的，那就意味着任意样本不会落在分离平面上，因此做这个修改也不影响求解。

如下图，实线为分离平面$x_i^T \beta + \beta_0 = 0$ ，两条虚线则为 $x_i^T \beta + \beta_0 = \pm 1$ ，虚线到实线的距离即为边距 $1/||\beta||^2$。
![SVM](SVM-margin.jpg)

这个问题可以很容易地用拉格朗日乘子法求解，由于这个模型并不是真正意义上的支持向量机，再此就不多做求解，直接进一步改进模型之后再求解添加了正则化项的模型。

## 支持向量分类器 Support Vector Classifier
如果样本观测不是严格的线性可分，那么前面基于分离平面的最大边距分类器将训练样本完全准确划分成两类是不妥当的，分离平面对接近分离平面的样本点过于敏感，可能导致过拟合。而且，基于硬边界原则得到的分类器可能只有一个很小的边距，而我们对分类的置信度与样本点距离分离平面的距离有关。由此，我们考虑允许对训练样本的不完全准确的分类，以求获得更大的边距（Margin），那么可以将问题的约束写为
$$
\begin{array}{ccc}
\underset{\beta,\beta_0}{min}\ \dfrac{1}{2}||\beta||^2 + C\sum\limits_{i=1}^{p} \xi_i \\ \\
\mathrm{subject\ to}\ y_i(x_i^T \beta + \beta_0) \geq 1-\xi_i\ ,\ \xi_i\geq0 \ \ \forall i=1,...,n
\end{array}
$$

其中，$C \geq 0$，$\xi_1,...,\xi_n$为松弛变量，以允许单个的观测落到错误地分类，$\sum\limits_{i=1}^{p} \xi_i$可以看作错分的总量。
一旦我们解得 $\beta$ ，得到超平面 $x^T \beta + \beta_0=0$ ，就可以对测试样本观测 $x^*$ 分类。与最大间隔分类器一样，也是利用 $f(x^*) = x^{*T} \beta + \beta_0$ 的符号分类。

最终得到的$\xi_i$的含义是，若$\xi_i>0$，那么意味着第 $i$ 个训练样本的位置在边距以内（violated the margin）；若$\xi_i>1$，那么意味着第i个样本在分离平面错误的一侧。加入这一项的目的是，如果错分少数几个样本，可以大幅增大边距 $M$ ，那么就容忍这些错分的样本，$C$ 越大，错分的代价就约高，倾向于尽可能正确地分离训练集。参数 $C$ 是可调节的参数，一般可以通过交叉验证（cross validation）确定。

问题求解得到的分类器称为支持向量分类器（support vector classifier），也称为软边界的支持向量机（soft margin support vector machine），而哪些落在边界(Margin)上或者边界内的样本则称为支持向量（support vector），支持向量机的训练结果只取决于所有的支持向量，其他的向量称为非支持向量（non-support vector），对训练结果没有影响。

考虑问题的拉格朗日函数
$$
L(\beta_0,\beta,\varepsilon,a,b) =
\dfrac{1}{2}||\beta||^2 + C \sum_{i=1}^p  \xi_i^2 - \sum_{i=1}^n  a_i\left[y_i(\beta_0+\beta^T x_i)-1+\xi_i \right] - \sum_{i=1}^n b_i \xi_i
$$

将问题转化为拉格朗日对偶问题
$$
\underset{a,b}{max} \underset{\beta_0,\beta,\xi}{inf}\ L(\beta_0,\beta,\xi,a,b)\ \ \mathrm{subject\ to}\ a,b \geq 0
$$

$L$分别对$\beta_0,\beta,\xi$求导
$$
\begin{array}{ccc}
\dfrac{\partial L}{\partial \beta_0} = 0 &\Rightarrow& \sum_{i=1}^n a_i y_i = 0 \\ \\
\dfrac{\partial L}{\partial \beta} = 0 &\Rightarrow& \beta = \sum_{i=1}^n a_i y_i x_i \\ \\
\dfrac{\partial L}{\partial \xi_i} = 0 &\Rightarrow& a_i + b_i = 0
\end{array}
$$

可以看到 $\beta,b$ 都可以由 $a,x,y$ 来表示，而$a_i + b_i = 0$则允许我们消去$L(\beta_0,\beta,\varepsilon,a,b)$ 中的$\varepsilon$，拉格朗日对偶问题化为
$$
\begin{array}{lll}
\hat{a} = \underset{a}{argmax} \left[ \sum_{i=1}^n  a_i - \sum_{i,j=1}^n  a_i a_j y_i x_i^T x_j \right]  \\ \\
subject\ to\ \sum_{i=1}^n a_i y_i = 0 ,\ 0\leq a_i \leq C \ for i=1,...,n
\end{array}
$$

又由解$\hat{a}$可以得到
$$ \hat{\beta} = \sum_{i=1}^n \hat{a} y_i x_i $$

以及
$$ \hat{\beta_0} = y_i - \sum_{j:\hat{a}>0} \hat{a} y_j x_i^T x_j $$

如果再考虑KKT条件的互补充实条件（complementary slackness）
$$ a_i(m_i-1+\varepsilon_i)=0\ \ and\ \ \beta_i \varepsilon_i=0 ,\ \forall i=1,...,n $$

其中 $m_i=y_i(\beta_0 + \beta x_i)$ 为第 $i$ 个样本的边距，又$a_i+b_i = 0$，可以总结得：
1. $a_i = 0$，说明$m_i > 0$：第i个样本$x_i$在边界上，但被正确地分类；
2. $0 < a_i < C$，说明$m_i \leq 1$：第i个样本$x_i$在边界上，如果$\varepsilon_i > 1$ 且$m_i \leq 1$，那么第i个样本被分错；
3. $m_i > 1$，说明$a_i=0$：即xi在正确的分类里，对边界没有影响；
4. $m_i < 1$，说明$a_i=C$：即xi在错误的分类里，对边界有影响。

从这里的式子可以知道，使得 $a_i > 0$ 的样本 $x_i$ 为支持向量（support vector），对训练结果有影响，其他的离分离平面较远的样本则是对训练结果没有影响的非支持向量（non-support vector）。

SMO求解