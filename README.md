# How to segment 𝜇-CT scans

This repository is a draft of a framework for segmenting 𝜇-CT scans. The
science of computational imaging has made a great deal of progress in the past
decades. There are a lot of great algorithms in scientific papers that can make
segmentation easy even in 3D CT scans that are noisy and have low contrast
between phases. Unfortunately, I don't see these scientific advances being used
in practice. Usually, people try to denoise the image using various techniques
(including ANNs) and then feed the output into ilastik or a similar piece of
software [[1]](#1). The idea that you have to make an image look nice with
denoising before segmentation is a misconception. It's not necessary with the
right algorithm. Perhaps the slow adoption of the advanced algorithms is due to
the fact that the papers contain very heavy math and are difficult to
understand. Whatever the reason, this repository aims to bridge the gap and show
how the right choice of algorithms can make the problem relatively easy and that
you don't need to denoise the image first.

Let's take a dataset from [[1]](#1). It's publicly available at Digital Porous
Media Portal [[2]](#2). The authors had a low-quality (LQ) scan to experiment
with, and a high-quality (HQ) scan. They are published as 1426 TIFF images each,
sprawling across 286 HTML pages. To download the LQ dataset without clicking
1426 links, you may issue the following command
```shell
curl --parallel --parallel-max 10 --remote-name-all -f --retry 3 "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-395/Glass%20(borosilicate)%20Core/LQ%20Dataset/Fast_32Bit_Registered[0000-1425].tif"
```
Then you can assemble them into one TIFF file with
```shell
tiffcp -c none -8 Fast_32Bit_Registered*.tif Fast_32Bit_Registered.tif
```

The scans represent super-critical CO<sub>2</sub> (scCO<sub>2</sub>) and brine
inside a glass (borosilicate) filter. The LQ dataset was intentionally measured
with low exposure to make it extremely noisy. The contrast between
scCO<sub>2</sub> and brine is comparable to the noise standard deviation. Plus,
the image suffers from a so-called "partial volume" effect: the interfaces
between phases are blurred and, therefore, voxels on the boundaries between
phases represent both neighbors. Below is a 3D cut through the central axis of
the CT scan. The scan axis is horizontal.

<p align="center">
  <picture>
    <img alt="SLice from the original LQ dataset" src="docs/DRP-395-LQ, slice-400 (𝜆 = 000.000).png" width="670px">
  </picture>
</p>

We can segment the above image without preliminary denoising using a maximum a
posteriori (MAP) estimator with the Potts model prior. In plain English, it
means that we will make an assumption (a prior) about our data: without noise,
our data image should consist of regions of uniform intensity. This type of
assumption is called the Potts model. Then we will try to maximize our prior
together with the likelihood. In general form, it can be expressed as

<a name="eq-potts"></a>
```math
    \min_{\left\{ E_ i\right\}^k_{i = 1}}{\left[
        \frac{1}{2}\sum_{i = 1}^k{
            \mathop{\mathrm{Per}}\nolimits{\left(E_i; \Omega\right)}
        } +
        \sum_{i = 1}^{k}{
            \int_{E_i}{g_{i}\left(x\right) dx}
        } \right]
    }
    \text{,}
    \qquad\text{(1)}
```

where $`\lbrace E_i\rbrace_{i=1}^{k}`$ is a partition of an open set $`\Omega
\subset \mathbb{R}^d`$, $`d \geq 2`$, into $`k`$ sets: $`E_i \cap E_j =
\emptyset`$ if $`i \neq j`$, and $`\bigcup_{i=1}^{k} E_i = \Omega`$ (up to
Lebesgue-negligible sets) [[3]](#3). $`g_i`$ in our case is derived from the MLE
for the Gaussian noise: $`g_i(x) = \lambda (c_i - u(x))^2`$, where $`c_i`$ is a
"label", the image intensity level for the phase $`i`$, $u(x)$ is our image, and
$\lambda$ is the weight we assign to our data.

The optimization problem ([1](#eq-potts)) is not easy to solve. Chambolle et al.
(2012) suggested a tight convex relaxation of this problem [[3]](#3). In other
words, they simplified the functional in ([1](#eq-potts)) just enough to make it
relatively easy to solve. They introduce the following notations. $\Xi$ denotes
$k$ $d$-dimensional vectors for every voxel, where $k$ is the number of labels
or phases in our CT scan, and $d = 3$ is the dimension of our data.
$\bar{\mathbf{V}}$ and $\mathbf{V}$ each represent $k$ values for each voxel. We
want to find $\mathbf{V}$. Its $k$ values are nonnegative numbers $\in [0, 1]$
that sum up to 1. The maximum number defines the label of the voxel. Chambolle
et al. (2012) build the following iterative procedure. First, they initialize
$\Xi^0 = 0$, $\bar{\mathbf{V}}^0 = \mathbf{V}^0 = 0$. Then they iteratively
update these values for each voxel:

```math
    \begin{aligned}
        \Xi^{n + 1} = & \Pi_{K}\left(
            \Xi^{n} + \tau'\left(\nabla^h{\bar{\mathbf{V}}^n}\right)
        \right)\text{,}\\
        \mathbf{V}^{n + 1} = & \Pi_{S}\left(
            \mathbf{V}^{n} + \tau \left(\left(\mathop{\mathrm{div}}\nolimits^h{\Xi^{n + 1}}\right) - \mathbf{G}^{h}\right)
        \right)\text{,}\\
        \bar{\mathbf{V}}^{{n + 1}} = & 2 \mathbf{V}^{n + 1} - \mathbf{V}^n
        \text{.}
    \end{aligned}

```
$\Pi_{S}$ is a projection of $k$ values on a simplex, i.e. $`v_i \in [0, 1]`$
and $`\sum_{i = 1}^k{v_i} = 1`$. $\Pi_{K}$ is a projection of $k$
$d$-dimensional vectors on set $K$, defined as

```math
    K = \left\{
        \mathbf{q} = \left(q_1, \ldots, q_k\right)^{T} \in \mathbb{R}^{k \times d}:
            \left\lvert q_i - q_j\right\rvert \leqslant 1 \ \forall i < j
    \right\}
    \text{.}
```

$h$ superscript denotes discretization. In particular, $\mathbf{G}^{h}$ is our
MLE potential for each voxel.

Note that both $\bar{\mathbf{V}}^n$ and $\mathbf{V}^n$ always sum up to 1
(except for the first step, and always if we initialize on the simplex), and
$\Xi^{n}$ always sum up to a zero vector for each voxel. Finally, $\tau$ and
$\tau'$ are step sizes. The algorithm converges as long as $\tau \tau' < 1 / 12$
(for $d = 3$).

The only question that remains to answer is how to find our labels $c_i$.

## How to find labels

## References

<a name="1"></a>
[1] Tawfik, M.S., Adishesha, A.S., Hsi, Y., Purswani, P., Johns, R.T., Shokouhi,
P., Huang, X., and Karpyn, Z.T. (2022) “Comparative study of traditional and
deep-learning denoising approaches for image-based petrophysical
characterization of porous media,” _Frontiers in Water_, 3, available:
https://doi.org/10.3389/frwa.2021.800369.

<a name="2"></a>
[2] Tawfik, M.S., Adishesha, A.S., Hsi, Y., Purswani, P., Johns, R.T., Shokouhi,
P., Huang, X., and Karpyn, Z.T. (2021) scCO<sub>2</sub>-Brine-Glass Dataset for
Comparing Image Denoising Algorithms. [online], available:
https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-395
[accessed 26 Sept 2026].

<a name="3"></a>
[3] Chambolle, A., Cremers, D., and Pock, T. (2012) “A convex approach to minimal
partitions,” SIAM Journal on Imaging Sciences, 5(4), 1113–1158, available:
https://doi.org/10.1137/110856733.