好的，下面是翻译后的内容，行内公式用 $$ 表示，行间公式用 $$ $$ 表示：

---

### 介绍

在之前的博客文章中，我们讨论了学习 $$ p(x) $$ 的两种方法：自回归模型 (ARMs) 和基于流的模型（简称流）。ARMs 和流都直接建模似然函数，即要么通过分解分布并对条件分布 $$ p(x_d | x_{<d}) $$ 进行参数化（如 ARMs），要么通过利用可逆变换（神经网络）作为变量变换公式（如流）。现在，我们将讨论第三种引入潜在变量的方法。

让我们简要讨论以下场景。我们有一组马的图像。我们想学习 $$ p(x) $$，例如，生成新图像。在此之前，我们可以问自己，如何生成一匹马，或者换句话说，如果我们是这样的生成模型，我们会怎么做。也许我们首先会勾勒出马的整体轮廓、大小和形状，然后添加蹄子，填充头部的细节，给它上色，等等。最后，我们可能还会考虑背景。一般来说，我们可以说数据中有一些因素（例如轮廓、颜色、背景）对于生成一个对象（这里是马）至关重要。一旦我们决定了这些因素，我们就可以通过添加细节来生成它们。我不想深入哲学/认知讨论，但我希望我们都同意，当我们画东西时，这或多或少是我们生成画作的过程。

我们现在用数学来表达这个生成过程。即，我们有我们感兴趣的高维对象 $$ x \in X^D $$（例如，对于图像，$$ X \in \{0, 1, \ldots, 255\} $$），以及低维潜在变量 $$ z \in Z^M $$（例如，$$ Z = \mathbb{R} $$），我们可以称之为数据中的隐含因素。在数学上，我们可以将 $$ Z^M $$ 视为一个低维流形。然后，生成过程可以表达如下：

$$
z \sim p(z) \quad \text{(图1，红色)}
$$

$$
x \sim p(x|z) \quad \text{(图1，蓝色)}
$$

通俗地说，我们首先采样 $$ z $$（例如，我们想象马的大小、形状和颜色），然后创建一幅包含所有必要细节的图像，即我们从条件分布 $$ p(x|z) $$ 中采样 $$ x $$。人们可能会问，我们在这里是否需要概率，但试着至少两次创造出完全相同的图像。由于许多外部因素，这几乎是不可能的。这就是概率论如此美妙的原因，它使我们能够描述现实！

**图1**. 一个展示潜在变量模型和生成过程的图示。注意低维流形（这里是2D）嵌入在高维空间中（这里是3D）。

潜在变量模型背后的想法是引入潜在变量 $$ z $$，并将联合分布分解如下：

$$
p(x,z) = p(x|z)p(z)
$$

这自然地表达了上述的生成过程。然而，对于训练，我们只能访问 $$ x $$。因此，根据概率推断，我们应该对未知数 $$ z $$ 进行求和（或边缘化）。因此，边际似然函数如下：

$$
p(x) = \int p(x|z)p(z) \, dz.
$$

现在一个自然的问题是如何计算这个积分。一般来说，这是一个困难的任务。有两个可能的方向。首先，积分是可处理的。在我们跳入第二种利用特定近似推断的方法之前，我们将简要讨论它，即变分推断。

### 概率主成分分析 (pPCA)：一种线性高斯潜在变量模型

让我们讨论以下情况：

我们仅考虑连续随机变量，即 $$ z \in \mathbb{R}^M $$ 和 $$ x \in \mathbb{R}^D $$。

$$ z $$ 的分布是标准高斯分布，即 $$ p(z) = N(z|0,I) $$。

$$ z $$ 和 $$ x $$ 之间的依赖关系是线性的，我们假设有高斯加性噪声：

$$
x = Wz + b + \epsilon,
$$

其中 $$ \epsilon \sim N(\epsilon|0,\sigma^2I) $$。高斯分布的性质导致：

$$
p(x|z) = N(x|Wz+b,\sigma^2I).
$$

该模型被称为概率主成分分析 (pPCA)（Tipping & Bishop, 1999）。

然后，我们可以利用两个正态分布随机变量的线性组合的性质显式计算积分（Bishop, 2006）：

$$
p(x) = \int p(x|z) p(z) \, dz = \int N(x|Wz+b,\sigma I) N(z|0,I) \, dz = N(x|b, WW^\top + \sigma^2 I).
$$

现在，我们可以计算边际似然函数的对数 $$ \ln p(x) $$！我们参考（Tipping & Bishop, 1999；Bishop, 2006）以获取有关在 pPCA 模型中学习参数的更多详细信息。此外，pPCA 的一个有趣之处在于，由于高斯的性质，我们还可以计算 $$ z $$ 的真实后验：

$$
p(z|x) = N(M^{-1} W^\top (x-\mu), \sigma^{-2} M)
$$

其中：$$ M = W^\top W + \sigma^2 I $$。一旦我们找到最大化对数似然函数的 $$ W $$，并且矩阵 $$ W $$ 的维度在计算上是可处理的，我们就可以计算 $$ p(z|x) $$。这是一件大事！为什么？因为对于给定的观察 $$ x $$，我们可以计算潜在因素的分布！

我认为，概率主成分分析是一个极其重要的潜在变量模型，原因有两个。首先，我们可以手动计算所有内容，因此，这是一个发展潜在变量模型直觉的绝佳练习。其次，它是一个线性模型，因此，好奇的读者应该已经感到一阵兴奋，并问自己以下问题：如果我们采用非线性依赖关系，会发生什么？如果我们使用其他分布而不是高斯分布，会发生什么？在这两种情况下，答案都是相同的：我们将无法精确计算积分，并且需要某种近似。无论如何，pPCA 是每个对潜在变量模型感兴趣的人都应该深入研究的模型，以建立对概率建模的直觉。

下面是翻译后的内容，行内公式用 $$ 表示，行间公式用 $$ $$ 表示：

---

### 变分自编码器：非线性潜在变量模型的变分推断

#### 模型与目标

让我们再看看积分，并考虑一个我们无法精确计算积分的一般情况。最简单的方法是使用蒙特卡罗近似：

$$
p(x) = \int p(x|z) p(z) \, dz = \mathbb{E}_{z \sim p(z)}[p(x|z)] \approx \frac{1}{K} \sum_k p(x|z_k)
$$

在最后一行中，我们使用来自潜在变量先验的样本，$$ z_k \sim p(z) $$。这种方法相对简单，由于我们的计算能力增长迅速，我们可以在合理的时间内采样许多点。然而，正如我们从统计学中所知，如果 $$ z $$ 是多维的，并且 $$ M $$ 相对较大，我们就陷入了维数灾难的陷阱，为了正确覆盖空间，样本数量会随 $$ M $$ 指数增长。如果我们样本过少，那么近似就会非常差。

我们可以使用更先进的蒙特卡罗技术（Andrieu, 2003），但它们仍然受到维数灾难相关问题的困扰。另一种方法是应用变分推断（Jordan et al., 1999）。让我们考虑一个由参数 $$ \phi $$ 描述的变分分布族，$$ \{q_\phi(z)\}_{\phi} $$。例如，我们可以考虑均值和方差为 $$ \phi = \{\mu, \sigma^2\} $$ 的高斯分布。我们知道这些分布的形式，并且假设它们对所有 $$ z \in Z^M $$ 分配非零的概率质量。那么，边际分布的对数可以近似为：

$$
\ln p(x) = \ln \int p(x|z) p(z) \, dz = \ln \int q_\phi(z) q_\phi(z) p(x|z) p(z) \, dz = \ln \mathbb{E}_{z \sim q_\phi(z)}[p(x|z) p(z) q_\phi(z)] \geq \mathbb{E}_{z \sim q_\phi(z)}[\ln [p(x|z) p(z) q_\phi(z)]] 
$$

$$
= \mathbb{E}_{z \sim q_\phi(z)}[\ln p(x|z)] - \mathbb{E}_{z \sim q_\phi(z)}[\ln q_\phi(z) - \ln p(z)].
$$

在第四行中，我们使用了詹森不等式。

如果我们考虑一种摊销的变分后验，即对每个 $$ x $$ 使用 $$ q_\phi(z|x) $$ 而不是 $$ q_\phi(z) $$，那么我们得到：

$$
\ln p(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)] - \mathbb{E}_{z \sim q_\phi(z|x)}[\ln q_\phi(z|x) - \ln p(z)].
$$

摊销可能极其有用，因为我们训练一个单一模型（例如，具有某些权重的神经网络），并且它为给定输入返回一个分布的参数。从现在开始，我们将假设我们使用摊销的变分后验，但请记住，我们不必这样做！请查看（Kim et al., 2018），其中考虑了半摊销的变分推断。

最终，我们得到一个类似自编码器的模型，具有随机编码器 $$ q_\phi(z|x) $$ 和随机解码器 $$ p(x|z) $$。我们使用“随机”来强调编码器和解码器是概率分布，并强调与确定性自编码器的区别。这个模型，具有摊销的变分后验，被称为变分自编码器（Kingma & Welling, 2013；Rezende et al., 2014）。对数似然函数的下界被称为证据下界（ELBO）。

ELBO 的第一部分 $$ \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)] $$ 被称为（负）重构误差，因为 $$ x $$ 被编码为 $$ z $$ 然后再解码回去。ELBO 的第二部分 $$ \mathbb{E}_{z \sim q_\phi(z|x)}[\ln q_\phi(z|x) - \ln p(z)] $$ 可以看作是一种正则化项，并且它与库尔巴克-莱布勒散度（KL 散度）一致。请记住，对于更复杂的模型（例如，层次模型），正则化项可能无法解释为 KL 项。因此，我们更愿意使用“正则化项”这个术语，因为它更为一般。

#### ELBO 的不同视角

为了完整起见，我们还提供 ELBO 的不同推导，这将帮助我们理解为什么下界有时可能棘手：

$$
\ln p(x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x)] = \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(z|x) p(x) p(z|x)] = \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z) p(z) p(z|x)] 
$$

$$
= \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z) p(z) q_\phi(z|x) q_\phi(z|x)] = \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z) p(z) q_\phi(z|x) q_\phi(z|x) p(z|x)] 
$$

$$
= \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z) - \ln q_\phi(z|x) p(z) + \ln q_\phi(z|x) p(z|x)] 
$$

$$
= \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)] - KL[q_\phi(z|x) \| p(z)] + KL[q_\phi(z|x) \| p(z|x)].
$$

请注意，在上述推导中，我们使用了求和和乘积规则，并结合乘以 $$ 1 = q_\phi(z|x) q_\phi(z|x) $$，没有其他花招！请尝试自己一步一步复制这个推导。如果你理解这个推导，会极大地帮助你看到 VAE（以及一般潜在变量模型）可能存在的潜在问题。

一旦你分析了这个推导，让我们仔细看看它：

$$
\ln p(x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)] - KL[q_\phi(z|x) \| p(z)] \quad \text{ELBO} + KL[q_\phi(z|x) \| p(z|x)] \geq 0.
$$

最后一个组件 $$ KL[q_\phi(z|x) \| p(z|x)] $$ 测量变分后验和真实后验之间的差异，但我们不知道真实后验是什么！然而，由于库尔巴克-莱布勒散度总是大于或等于 0（根据其定义），因此我们得到 ELBO。我们可以将 $$ KL[q_\phi(z|x) \| p(z|x)] $$ 看作是 ELBO 和真实对数似然之间的差距。

美妙的是！但这为什么如此重要呢？如果我们取 $$ q_\phi(z|x) $$ 是 $$ p(z|x) $$ 的一个糟糕近似，那么 KL 项将会更大，即使 ELBO 被很好地优化，ELBO 和真实对数似然之间的差距也可能很大！通俗地说，如果我们选择了过于简单的后验，我们可能仍然会得到一个糟糕的 VAE。在这个上下文中，“糟糕”是什么意思？让我们看看图 2。如果 ELBO 是对数似然的松弛下界，那么 ELBO 的最优解可能与对数似然的解完全不同。我们稍后会评论如何处理这个问题，现在，意识到这个问题就足够了。

下面是翻译后的内容，行内公式用 $$ 表示，行间公式用 $$ $$ 表示：

---

### 图 2. ELBO 是对数似然的下界。因此，最大化 ELBO 的 $$ \hat{\theta} $$ 不一定与最大化 $$ \ln p(x) $$ 的 $$ \theta^* $$ 重合。ELBO 越松弛，这可能会偏差模型参数的最大似然估计。

#### VAE 的组成部分

让我们总结一下目前所知道的。首先，我们考虑一类摊销的变分后验 $$ \{q_\phi(z|x)\}_{\phi} $$，它们近似真实后验 $$ p(z|x) $$。我们可以将它们视为随机编码器。其次，条件似然 $$ p(x|z) $$ 可以看作是一个随机解码器。第三，最后一个组成部分 $$ p(z) $$ 是边际分布，也称为先验。最后，目标是 ELBO，作为对数似然函数的下界：

$$
\ln p(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)}[\ln p(x|z)] - \mathbb{E}_{z \sim q_\phi(z|x)}[\ln q_\phi(z|x) - \ln p(z)].
$$

要获得 VAE 的完整图景，还有两个问题待解答：

1. 如何参数化这些分布？
2. 如何计算期望值？毕竟，这些积分并没有消失！

#### 分布的参数化

正如你现在可能猜到的那样，我们使用神经网络来参数化编码器和解码器。但是，在使用神经网络之前，我们应该知道使用什么分布！幸运的是，在 VAE 框架中，我们几乎可以自由选择任何分布！然而，我们必须记住，它们应该对考虑的问题有意义。到目前为止，我们通过图像解释了一切，因此我们继续这个思路。如果 $$ x \in \{0, 1, \ldots, 255\}^D $$，那么我们不能使用正态分布，因为它的支持与离散值图像的支持完全不同。我们可以使用的一个可能分布是类别分布。现在我们有：

$$
p_\theta(x|z) = \text{Categorical}(x | \theta(z)),
$$

其中概率由一个神经网络 $$ \text{NN} $$ 给出，即 $$ \theta(z) = \text{softmax}(\text{NN}(z)) $$。神经网络 $$ \text{NN} $$ 可以是多层感知机、卷积神经网络、递归神经网络等。

潜在变量的分布选择取决于我们如何想要在数据中表达潜在因素。为了方便，通常将 $$ z $$ 视为连续随机变量的向量，即 $$ z \in \mathbb{R}^M $$。然后，我们可以为变分后验和先验使用高斯分布：

$$
q_\phi(z|x) \, p(z) = \mathcal{N}(z|\mu_\phi(x), \text{diag}[\sigma^2_\phi(x)]) = \mathcal{N}(z|0,I),
$$

其中 $$ \mu_\phi(x) $$ 和 $$ \sigma^2_\phi(x) $$ 是神经网络的输出，类似于解码器的情况。在实践中，我们可以有一个共享的神经网络 $$ \text{NN}(x) $$，输出 $$ 2M $$ 个值，然后将它们分成 $$ M $$ 个均值 $$ \mu $$ 和 $$ M $$ 个方差 $$ \sigma^2 $$。为了方便，我们考虑对角协方差矩阵。此外，这里我们采用标准高斯先验。我们稍后会对此进行评论。

#### 重参数化技巧

到目前为止，我们讨论了对数似然，最终得到了 ELBO。然而，计算期望值仍然存在问题，因为它包含一个积分！因此，问题是我们如何计算它，以及为什么这比没有变分后验的对数似然的 MC 近似更好。实际上，我们将使用 MC 近似，但现在，不是从先验 $$ p(z) $$ 中采样，而是从变分后验 $$ q_\phi(z|x) $$ 中采样。这更好吗？是的，因为变分后验通常在较小的区域内分配更多的概率质量，而不是先验。如果你玩弄你的 VAE，并检查方差，你会发现变分后验几乎是确定性的（这是否好还是坏仍然是一个开放的问题）。因此，我们应该得到更好的近似！但是，近似的方差仍然存在问题。简单来说，如果我们从 $$ q_\phi(z|x) $$ 中采样 $$ z $$，将它们代入 ELBO，并计算相对于神经网络参数 $$ \phi $$ 的梯度，那么梯度的方差仍然可能相当大！解决这个问题的一个可能方案，首先被统计学家注意到（例如，参见 Devroye, 1996），就是重参数化分布的思想。这个思想是意识到我们可以将随机变量表示为独立随机变量通过简单分布的基本变换（例如，算术运算、对数等）的组合。例如，如果我们考虑一个均值为 $$ \mu $$ 和方差为 $$ \sigma^2 $$ 的高斯随机变量 $$ z $$，以及一个独立随机变量 $$ \epsilon \sim \mathcal{N}(\epsilon|0,1) $$，则有：

$$
z = \mu + \sigma \cdot \epsilon.
$$

现在，如果我们开始从标准高斯分布中采样 $$ \epsilon $$，并应用上述变换，我们就得到了来自 $$ \mathcal{N}(z|\mu,\sigma) $$ 的样本！

### 图 3. 重参数化高斯分布的示例：我们将根据标准高斯分布的 $$ \epsilon $$ 缩放 $$ \sigma $$ 并用 $$ \mu $$ 平移。

如果你不记得这个统计学事实，或者你根本不相信我，可以写一段简单的代码来试试。实际上，这个思想可以应用于更多分布（Kingma & Welling, 2014）。

重参数化技巧可以应用于编码器 $$ q_\phi(z|x) $$。正如（Kingma & Welling, 2013；Rezende et al., 2014）所观察到的，我们可以通过使用这种高斯分布的重参数化显著降低梯度的方差。为什么？因为随机性来自独立源 $$ p(\epsilon) $$，而我们是相对于一个确定性函数（即神经网络）计算梯度，而不是随机对象。更好的是，由于我们使用随机梯度下降学习 VAE，训练期间只需采样一次 $$ z $$ 即可！

下面是翻译后的内容，行内公式用 $$ 表示，行间公式用 $$ $$ 表示：

---

### VAE 实战！

我们经历了很多理论和讨论，你可能会认为实现一个 VAE 不可能。然而，实际上它比看起来简单。让我们总结一下到目前为止所知道的内容，并关注非常具体的分布和神经网络。

首先，我们将使用以下分布：

$$
q_\phi(z|x) = \mathcal{N}(z|\mu_\phi(x),\sigma^2_\phi(x));
$$

$$
p(z) = \mathcal{N}(z|0,I);
$$

$$
p_\theta(x|z) = \text{Categorical}(x|\theta(z)).
$$

我们假设 $$ x_d \in X = \{0, 1, \ldots, L-1\} $$。

接下来，我们将使用以下网络：

#### 编码器网络：
$$
x \in X^D \rightarrow \text{Linear}(D, 256) \rightarrow \text{LeakyReLU} \rightarrow \text{Linear}(256, 2M) \rightarrow \text{split} \rightarrow \mu \in \mathbb{R}^M, \log \sigma^2 \in \mathbb{R}^M
$$

注意，最后一层输出 $$ 2M $$ 个值，因为我们必须有 $$ M $$ 个均值和 $$ M $$ 个（对数）方差。此外，方差必须是正的，因此，我们考虑方差的对数，因为它可以取实数值。结果是，我们不需要担心方差始终为正。

#### 解码器网络：
$$
z \in \mathbb{R}^M \rightarrow \text{Linear}(M, 256) \rightarrow \text{LeakyReLU} \rightarrow \text{Linear}(256, D \cdot L) \rightarrow \text{reshape} \rightarrow \text{softmax} \rightarrow \theta \in [0, 1]^{D \times L}
$$

由于我们对 $$ x $$ 使用类别分布，因此解码器网络的输出是概率。首先，最后一层必须输出 $$ D \cdot L $$ 个值，其中 $$ D $$ 是像素的数量，$$ L $$ 是像素可能值的数量。然后，我们必须将输出重塑为形状为 $$ (B, D, L) $$ 的张量，其中 $$ B $$ 是批大小。之后，我们可以应用 softmax 激活函数以获得概率。

最后，对于给定的数据集 $$ D = \{x_n\}_{n=1}^N $$，训练目标是 ELBO，我们使用来自变分后验的单个样本 $$ z_{\phi,n} = \mu_\phi(x_n) + \sigma_\phi(x_n) \odot \epsilon $$。我们必须记住，在几乎所有可用的包中，默认情况下是最小化，因此我们必须取负号，即：

$$
-\text{ELBO}(D;\theta,\phi) = \sum_{n=1}^N -\left\{\ln \text{Categorical}(x_n|\theta(z_{\phi,n})) + \left[\ln \mathcal{N}(z_{\phi,n}|\mu_\phi(x_n), \sigma^2_\phi(x_n)) + \ln \mathcal{N}(z_{\phi,n}|0,I)\right]\right\}
$$

正如你所看到的，整个数学归结为一个相对简单的学习过程：

1. 获取 $$ x_n $$，并应用编码器网络以获得 $$ \mu_\phi(x_n) $$ 和 $$ \ln \sigma^2_\phi(x_n) $$。
2. 通过应用重参数化技巧计算 $$ z_{\phi,n} $$，即 $$ z_{\phi,n} = \mu_\phi(x_n) + \sigma_\phi(x_n) \odot \epsilon $$，其中 $$ \epsilon \sim \mathcal{N}(0,I) $$。
3. 将 $$ z_{\phi,n} $$ 输入解码器网络以获得概率 $$ \theta(z_{\phi,n}) $$。
4. 插入 $$ x_n $$、$$ z_{\phi,n} $$、$$ \mu_\phi(x_n) $$ 和 $$ \ln \sigma^2_\phi(x_n) $$ 计算 ELBO。

现在，所有组件都可以转化为代码！有关完整实现，请查看 [链接]。在这里，我们只关注 VAE 模型的代码。我们在注释中提供详细信息。我们将代码分为四个类：Encoder、Decoder、Prior 和 VAE。它可能看起来有些多余，但它可能有助于你将 VAE 视为三个部分的组合，从而更好地理解整个方法。



```
class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        
        # The init of the encoder network.
        self.encoder = encoder_net
    
    # The reparameterization trick for Gaussians.
    @staticmethod
    def reparameterization(mu, log_var):
        # The formulat is the following:
        # z = mu + std * epsilon
        # epsilon ~ Normal(0,1)
        
        # First, we need to get std from log-variance.
        std = torch.exp(0.5*log_var)
        
        # Second, we sample epsilon from Normal(0,1).
        eps = torch.randn_like(std)
        
        # The final output
        return mu + std * eps
    
    # This function implements the output of the encoder network (i.e., parameters of a Gaussian).
    def encode(self, x):
        # First, we calculate the output of the encoder netowork of size 2M.
        h_e = self.encoder(x)
        # Second, we must divide the output to the mean and the log-variance.
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e
    
    # Sampling procedure.
    def sample(self, x=None, mu_e=None, log_var_e=None):
        #If we don't provide a mean and a log-variance, we must first calcuate it:
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        # Or the final sample
        else:
        # Otherwise, we can simply apply the reparameterization trick!
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-var can`t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

    # This function calculates the log-probability that is later used for calculating the ELBO.
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        # If we provide x alone, then we can calculate a corresponsing sample:
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
        # Otherwise, we should provide mu, log-var and z!
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-var and z can`t be None!')
        
        return log_normal_diag(z, mu_e, log_var_e)
    
    # PyTorch forward pass: it is either log-probability (by default) or sampling.
    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)
            
class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_vals=None):
        super(Decoder, self).__init__()
        
        # The decoder network.
        self.decoder = decoder_net
        # The distribution used for the decoder (it is categorical by default, as discussed above).
        self.distribution = distribution
        # The number of possible values. This is important for the categorical distribution.
        self.num_vals=num_vals
    
    # This function calculates parameters of the likelihood function p(x|z)
    def decode(self, z):
        # First, we apply the decoder network.
        h_d = self.decoder(z)
        
        # In this example, we use only the categorical distribution...
        if self.distribution == 'categorical':
            # We save the shapes: batch size
            b = h_d.shape[0]
            # and the dimensionality of x.
            d = h_d.shape[1]//self.num_vals
            # Then we reshape to (Batch size, Dimensionality, Number of Values).
            h_d = h_d.view(b, d, self.num_vals)
            # To get probabilities, we apply softmax.
            mu_d = torch.softmax(h_d, 2)
            return [mu_d]
        # ... however, we also present the Bernoulli distribution. We are nice, aren't we?
        elif self.distribution == 'bernoulli':
            # In the Bernoulli case, we have x_d \in {0,1}. Therefore, it is enough to output a single probability,
            # because p(x_d=1|z) = \theta and p(x_d=0|z) = 1 - \theta
            mu_d = torch.sigmoid(h_d)
            return [mu_d]
        
        else:
            raise ValueError('Either `categorical` or `bernoulli`')
    
    # This function implements sampling from the decoder.
    def sample(self, z):
        outs = self.decode(z)

        if self.distribution == 'categorical':
            # We take the output of the decoder
            mu_d = outs[0]
            # and save shapes (we will need that for reshaping). 
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            # Here we use reshaping
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            # Eventually, we sample from the categorical (the built-in PyTorch function).
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        elif self.distribution == 'bernoulli':
            # In the case of Bernoulli, we don't need any reshaping
            mu_d = outs[0]
            # and we can use the built-in PyTorch sampler!
            x_new = torch.bernoulli(mu_d)
        
        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return x_new
    
    # This function calculates the conditional log-likelihood function.
    def log_prob(self, x, z):
        outs = self.decode(z)

        if self.distribution == 'categorical':
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)
            
        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)
            
        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return log_p
    
    # The forward pass is either a log-prob or a sample.
    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(x)
            
# The current implementation of the prior is very simple, namely, it is a standard Gaussian.
# We could have used a built-in PuTorch distribution. However, we didn't do that for two reasons:
# (i) It is important to think of the prior as a crucial component in VAEs.
# (ii) We can implement a learnable prior (e.g., a flow-based prior, VampPrior, a muxture of distributions).
class Prior(nn.Module):
    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, z):
        return log_standard_normal(z)
        
class VAE(nn.Module):
    def __init__(self, encoder_net, decoder_net, num_vals=256, L=16, likelihood_type='categorical'):
        super(VAE, self).__init__()

        print('VAE by JT.')

        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(distribution=likelihood_type, decoder_net=decoder_net, num_vals=num_vals)
        self.prior = Prior(L=L)

        self.num_vals = num_vals

        self.likelihood_type = likelihood_type

    def forward(self, x, reduction='avg'):
        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # ELBO
        RE = self.decoder.log_prob(x, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)

        if reduction == 'sum':
            return -(RE + KL).sum()
        else:
            return -(RE + KL).mean()

    def sample(self, batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)
        
# Examples of neural networks used for parameterizing the encoder and the decoder.

# Remember that the encoder outputs 2 times more values because we need L means and L log-variances for a Gaussian.
encoder = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                        nn.Linear(M, M), nn.LeakyReLU(),
                        nn.Linear(M, 2 * L))

# Here we must remember that if we use the categorical distribution, we must output num_vals per each pixel.
decoder = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                        nn.Linear(M, M), nn.LeakyReLU(),
                        nn.Linear(M, num_vals * D))
```

下面是翻译后的内容，行内公式用 $$ 表示，行间公式用 $$ $$ 表示：

---

### 完美！现在我们准备运行完整的代码（请查看：[链接]）。在训练我们的 ARM 之后，我们应该获得类似于以下结果：

**图 4. 训练结果示例：**
- A 随机选择的真实图像。
- B 来自 VAE 的无条件生成。
- C 训练过程中的验证曲线。

### VAEs 的典型问题

VAEs 是一种非常强大的模型类别，主要由于其灵活性。与基于流的模型不同，它们不需要神经网络的可逆性，因此我们可以为编码器和解码器使用任何任意架构。与 ARMs 相比，它们学习低维的数据表示，我们可以控制瓶颈（即潜在空间的维度）。然而，它们也面临几个问题。除了之前提到的问题（即需要有效的积分估计，过于简单的变分后验导致 ELBO 和对数似然函数之间存在差距）外，潜在的问题包括：

让我们看看 ELBO 和正则化项。对于不可训练的先验分布，如标准高斯分布，当 $$ \forall x, q_\phi(z|x) = p(z) $$ 时，正则化项将被最小化。这可能发生在解码器非常强大，以至于将 $$ z $$ 视为噪声时，例如，当解码器被表示为 ARM（Alemi et al., 2018）。这个问题被称为后验崩溃（posterior collapse）（Bowman et al., 2015）。

另一个问题与聚合后验 $$ q_\phi(z) = \frac{1}{N} \sum_n q_\phi(z|x_n) $$ 和先验 $$ p(z) $$ 之间的不匹配有关。想象一下我们有标准高斯先验和聚合后验（即对所有训练数据的变分后验的平均值）。结果是，在某些区域，先验分配高概率，但聚合后验分配低概率，反之亦然。然后，从这些“空洞”中采样提供不现实的潜在值，解码器生成的图像质量很低。这个问题被称为空洞问题（hole problem）（Rezende & Viola, 2018）。

我们想讨论的最后一个问题更加普遍，实际上影响所有深度生成模型。正如 (Nalisnick et al., 2018) 中指出的，深度生成模型（包括 VAEs）未能正确检测到分布外的示例。分布外数据点是与模型训练的分布完全不同的示例。例如，假设我们的模型是在 MNIST 上训练的，那么 FashionMNIST 的示例就是分布外的。因此，一个直觉是，经过适当训练的深度生成模型应该给分布内示例分配高概率，而给分布外点分配低概率。不幸的是，正如 (Nalisnick et al., 2018) 所示，情况并非如此。分布外问题仍然是深度生成建模中的主要未解决问题之一。



下面是翻译后的内容，行内公式用 $$ 表示，行间公式用 $$ $$ 表示：

---

### 还有很多很多其他研究！

有大量的论文扩展了 VAEs 并将其应用于许多问题。下面我们将列出一些精选论文，并仅触及这一主题的广泛文献！

#### 使用重要性加权估计对数似然

正如我们多次提到的，ELBO 是对数似然的下界，实际上不应作为对数似然的良好估计。在 (Burda et al., 2015; Rezende et al., 2014) 中，提倡了一种重要性加权程序，以更好地近似对数似然，具体为：

$$
\ln p(x) \approx \ln \left( \frac{1}{K} \sum_{k=1}^{K} p(x|z_k) q_\phi(z_k|x) \right),
$$

其中 $$ z_k \sim q_\phi(z_k|x) $$。注意，取对数是在期望值外部进行的。如 (Burda et al., 2015) 所示，使用足够大的 $$ K $$ 进行重要性加权能提供对数似然的良好估计。在实践中，$$ K $$ 通常取为 512 或更多，具体取决于计算预算。

#### 增强 VAEs：更好的编码器

在引入 VAE 的概念后，许多论文专注于提出灵活的变分后验族。最突出的方向是利用条件流模型 (van den Berg et al., 2018; Hoogeboom et al., 2020; Kingma et al., 2016; Rezende & Mohamed, 2015; Tomczak & Welling, 2016; Tomczak & Welling, 2017)。

#### 增强 VAEs：更好的解码器

VAEs 允许使用任何神经网络来参数化解码器。因此，我们可以使用全连接网络、全卷积网络、ResNets 或 ARMs。例如，在 (Gulrajani et al., 2016) 中，使用基于 PixelCNN 的解码器在 VAE 中进行了应用。

#### 增强 VAEs：更好的先验

如前所述，如果聚合后验与先验之间存在较大不匹配，这可能是一个严重的问题。有许多论文试图通过使用模拟聚合后验的多模态先验（称为 VampPrior）（Tomczak & Welling, 2018），或使用基于流的先验（例如 (Gatopoulos & Tomczak, 2020)）、基于 ARM 的先验（Chen et al., 2016），或使用重采样的想法（Bauer & Mnih, 2019）来缓解这个问题。

#### 扩展 VAEs

在这里，我们介绍了 VAEs 的无监督版本。然而，实际上没有限制，我们可以引入标签或其他变量。在 (Kingma et al., 2014) 中，提出了一种半监督 VAE。这一思想进一步扩展到公平表示的概念中 (Louizos et al., 2015)。在 (Ilse et al., 2020) 中，作者提出了一种特定的潜在表示，允许 VAEs 中的领域泛化。在 (Blundell et al., 2015) 中，变分推断和重参数化技巧被应用于贝叶斯神经网络。虽然这篇论文不一定引入 VAE，但提供了一种类似 VAE 的处理贝叶斯神经网络的方法。

#### VAEs 在非图像数据中的应用

在这篇文章中，我解释了所有与图像相关的内容。然而，这并没有限制！在 (Bowman et al., 2015) 中，提出了一种 VAE 用于处理序列数据（例如，文本）。编码器和解码器由 LSTM 参数化。VAEs 的一个有趣应用也在 (Jin et al., 2018) 中呈现，其中 VAEs 被用于分子图的生成。在 (Habibian et al., 2019) 中，作者提出了一种 VAE 风格的视频压缩方法。

#### 不同的潜在空间

通常，考虑的是欧几里得潜在空间。然而，VAE 框架允许我们考虑其他空间。例如，在 (Davidson et al., 2018; Davidson et al., 2019) 中使用了超球面潜在空间，在 (Mathieu et al., 2019) 中使用了超曲率潜在空间。

#### 后验崩溃

有许多想法被提出以应对后验崩溃。例如，(He et al., 2019) 提出比解码器更频繁地更新变分后验。在 (Dieng et al., 2019) 中，通过引入跳跃连接来避免后验崩溃，提出了一种新的解码器架构。

#### 目标的多种视角

VAE 的核心是 ELBO。然而，我们可以考虑不同的目标。例如，(Dieng et al., 2017) 提出了基于卡方散度的对数似然上界（CUBO）。在 (Alemi et al., 2018) 中，提出了关于 ELBO 的信息论视角。 (Higgins et al., 2016) 引入了 $$ \beta $$ -VAE，其中正则化项由一个调节因子 $$ \beta $$ 加权。尽管目标不对应于对数似然的下界。

#### 确定性正则化自编码器

我们可以将 VAE 和前述目标视为带有随机编码器和随机解码器的自编码器的正则化版本。 (Ghosh et al., 2020) 将 VAEs 中的所有随机性“剥离”，并指出确定性正则化自编码器与 VAEs 之间的相似性，同时强调了 VAEs 的潜在问题。此外，他们巧妙地指出，即使是确定性编码器，由于经验分布的随机性，我们也可以将模型拟合到聚合后验。因此，确定性（正则化）自编码器可以通过从我们的模型 $$ p_\lambda(z) $$ 中采样，并确定性地将 $$ z $$ 映射到可观察空间 $$ x $$，转变为生成模型。在我看来，这一方向应该进一步探索，一个重要的问题是我们是否真的需要任何随机性。

#### 分层 VAEs

最近，有许多具有深层分层潜在变量结构的 VAEs 取得了显著成果！其中最重要的包括 BIVA (Maaløe et al., 2019)、NVA (Vahdat & Kautz, 2020) 和非常深的 VAEs (Child, 2020)。在 (Gatopoulos & Tomczak, 2020) 中，呈现了关于深层分层 VAE 的另一个有趣视角，此外，还使用了一系列确定性函数。

#### 对抗自编码器

(Makhzani et al., 2015) 中提出了对 VAEs 的另一种有趣视角。由于学习聚合后验作为先验是一些论文（例如 (Tomczak & Welling, 2018)）中提到的重要组成部分，因此一种不同的方法是用对抗损失训练先验。此外，(Makhzani et al., 2015) 提出了各种想法，说明自编码器如何从对抗学习中受益。





1. Alemi, A., Poole, B., Fischer, I., Dillon, J., Saurous, R. A., & Murphy, K. (2018, July). Fixing a broken ELBO. International Conference on Machine Learning (pp. 159-168). PMLR.

2. Andrieu, C., De Freitas, N., Doucet, A., & Jordan, M. I. (2003). An introduction to MCMC for machine learning. Machine learning, 50(1-2), 5-43.

3. Bauer, M., & Mnih, A. (2019). Resampled priors for variational autoencoders. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 66-75). PMLR.

4. van den Berg, R., Hasenclever, L., Tomczak, J. M., & Welling, M. (2018). Sylvester normalizing flows for variational inference. UAI 2018.

5. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

6. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015, July). Weight uncertainty in neural networks. In Proceedings of the 32nd International Conference on International Conference on Machine Learning-Volume 37 (pp. 1613-1622).

7. Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2015). Generating sentences from a continuous space. arXiv preprint arXiv:1511.06349.

8. Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance weighted autoencoders. arXiv preprint arXiv:1509.00519.

9. Chen, X., Kingma, D. P., Salimans, T., Duan, Y., Dhariwal, P., Schulman, J., Sutskever, I., & Abbeel, P. (2016). Variational lossy autoencoder. arXiv preprint arXiv:1611.02731.

10. Child, R. (2020). Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images. arXiv preprint arXiv:2011.10650.

11. Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T., & Tomczak, J. M. (2018). Hyperspherical variational auto-encoders. UAI 2018.

12. Davidson, T. R., Tomczak, J. M., & Gavves, E. (2019). Increasing Expressivity of a Hyperspherical VAE. arXiv preprint arXiv:1910.02912.

13. Devroye, L. (1996). Random variate generation in one line of code. In Proceedings Winter Simulation Conference (pp. 265-272). IEEE.

14. Dieng, A. B., Tran, D., Ranganath, R., Paisley, J., & Blei, D. (2017). Variational Inference via χ Upper Bound Minimization. In Advances in Neural Information Processing Systems (pp. 2732-2741).

15. Dieng, A. B., Kim, Y., Rush, A. M., & Blei, D. M. (2019). Avoiding latent variable collapse with generative skip models. In The 22nd International Conference on Artificial Intelligence and Statistics (pp. 2397-2405). PMLR.

16. Gatopoulos, I., & Tomczak, J. M. (2020). Self-Supervised Variational Auto-Encoders. arXiv preprint arXiv:2010.02014.

17. Ghosh, P., Sajjadi, M. S., Vergari, A., Black, M., & Schölkopf, B. (2020). From variational to deterministic autoencoders. ICLR.

18. Gulrajani, I., Kumar, K., Ahmed, F., Taiga, A. A., Visin, F., Vazquez, D., & Courville, A. (2016). Pixelvae: A latent variable model for natural images. arXiv preprint arXiv:1611.05013.

19. Habibian, A., Rozendaal, T. V., Tomczak, J. M., & Cohen, T. S. (2019). Video compression with rate-distortion autoencoders. In Proceedings of the IEEE International Conference on Computer Vision (pp. 7033-7042).

20. He, J., Spokoyny, D., Neubig, G., & Berg-Kirkpatrick, T. (2019). Lagging inference networks and posterior collapse in variational autoencoders. arXiv preprint arXiv:1901.05534.

21. Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2016). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.

22. Hoffman, M. D., & Johnson, M. J. (2016). Elbo surgery: yet another way to carve up the variational evidence lower bound. In Workshop in Advances in Approximate Bayesian Inference, NIPS (Vol. 1, p. 2).

23. Hoogeboom, E., Satorras, V. G., Tomczak, J. M., & Welling, M. (2020). The Convolution Exponential and Generalized Sylvester Flows. arXiv preprint arXiv:2006.01910.

24. Ilse, M., Tomczak, J. M., Louizos, C., & Welling, M. (2020). DIVA: Domain invariant variational autoencoders. In Medical Imaging with Deep Learning (pp. 322-348). PMLR.

25. Jin, W., Barzilay, R., & Jaakkola, T. (2018). Junction Tree Variational Autoencoder for Molecular Graph Generation. In International Conference on Machine Learning (pp. 2323-2332).

26. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine learning, 37(2), 183-233.

27. Kim, Y., Wiseman, S., Miller, A., Sontag, D., & Rush, A. (2018). Semi-amortized variational autoencoders. In International Conference on Machine Learning (pp. 2678-2687). PMLR.

28. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

29. Kingma, D., & Welling, M. (2014). Efficient gradient-based inference through transformations between bayes nets and neural nets. In International Conference on Machine Learning (pp. 1782-1790).

30. Kingma, D. P., Mohamed, S., Jimenez Rezende, D., & Welling, M. (2014). Semi-supervised learning with deep generative models. Advances in neural information processing systems, 27, 3581-3589.

31. Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M. (2016). Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems, 29, 4743-4751.

32. Louizos, C., Swersky, K., Li, Y., Welling, M., & Zemel, R. (2015). The variational fair autoencoder. arXiv preprint arXiv:1511.00830.

33. Maaløe, L., Fraccaro, M., Liévin, V., & Winther, O. (2019). Biva: A very deep hierarchy of latent variables for generative modeling. In Advances in neural information processing systems (pp. 6551-6562).

34. Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.

35. Mathieu, E., Le Lan, C., Maddison, C. J., Tomioka, R., & Teh, Y. W. (2019). Continuous hierarchical representations with poincaré variational auto-encoders. In Advances in neural information processing systems (pp. 12565-12576).

36. Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., & Lakshminarayanan, B. (2018). Do deep generative models know what they don't know?. arXiv preprint arXiv:1810.09136.

37. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. International Conference on Machine Learning (pp. 1278-1286).

38. Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. ICML 2015.

39. Rezende, D. J., & Viola, F. (2018). Taming vaes. arXiv preprint arXiv:1810.00597.

40. Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61(3), 611-622.

41. Tomczak, J. M., & Welling, M. (2016). Improving variational auto-encoders using householder flow. arXiv preprint arXiv:1611.09630.

42. Tomczak, J. M., & Welling, M. (2017). Improving variational auto-encoders using convex combination linear inverse autoregressive flow. arXiv preprint arXiv:1706.02326.

43. Tomczak, J., & Welling, M. (2018). VAE with a VampPrior. Artificial Intelligence and Statistics (pp. 1214-1223). PMLR.

44. Vahdat, A., & Kautz, J. (2020). NVAE: A deep hierarchical variational autoencoder. arXiv preprint arXiv:2007.03898.