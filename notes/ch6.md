# Clustering

## K-means clustering

Clustering ：

* Goal: choose “centers” as the representative intensities, and label every pixel according to which of these centers it is nearest to.

* Best cluster centers are those that minimize Sum of Square Distance (SSD) between all points and their nearest cluster center $c_i$:
  $$
  SSD = \sum_{cluster\ i}\sum_{x\in cluster\ i}(x-c_i)^2
  $$

Clustering for Summarization

* Goal: cluster to minimize variance in data given clusters
  $$
  c^*,\delta^*=\arg\min_{c,\delta}\frac{1}{N}\sum_{j}^{N}\sum_{i}^{K}\delta_{ij}(c_i-x_j)^2
  $$

**K-means clustering**

1. Initialize$(t=0)$: cluster center $c_1,\cdots,c_k$

2. Compute $\delta^t$: assign each point to the closest center

   * $\delta^t$ denotes the set of assignment for each $x_j$ to cluster $c_i$ at iteration $t$
     $$
     \delta^t = \arg\min_{\delta}\frac{1}{N}\sum_{j}^{N}\sum_{i}^{K}\delta_{ij}^{t-1}(c_i^{t-1}-x_j)^2
     $$

3. Compute $c^t$: update cluster centers as the mean of the points
   $$
   c^t = \arg\min_{c}\frac{1}{N}\sum_{j}^{N}\sum_{i}^{N}\delta_{ij}^t(c_i^{t-1}-x_j)^2
   $$

4. Update $t=t+1$, repeat Step $2-3$ till stopped.

**K-Means++**

1. Randomly choose first center
2. Pick new center with prob.proportional to $(x-c_i)^2$
3. Repeat until K centers.

Feature space:

* Depending on what we choose as the feature space, we can group pixels in different ways.
  * Grouping pixels based on intensity similarity
  * Grouping pixels based on color similarity
  * Grouping pixels based on texture similarity

<img src="ch6.assets/image-20210105103932361.png" alt="image-20210105103932361" style="zoom:50%;" />

## Mean-shift clustering

1. Initialize random seed, and window W
2. Calculate center of gravity (the “mean”) of W: $\sum_{x\in W}xH(x)$
3. Shift the search window to the mean
4. Repeat Step 2 until convergence

Mean-Shift Clustering/Segmentation:

* Find features (color, gradients, texture, etc)
* Initialize windows at individual pixel locations
* Perform mean shift for each window until convergence
* Merge windows that end up near the same “peak” or mode

<img src="ch6.assets/image-20210105111442600.png" alt="image-20210105111442600" style="zoom: 50%;" />

Problem： Computational Complexity

Method：

<img src="ch6.assets/image-20210106093001452.png" alt="image-20210106093001452" style="zoom: 50%;" />

**Technical Details**

Given $n$ data points $x_i\in \mathbb{R}^d$, the multivariate kernel density estimate using a radially symmetric kernel (e.g., Epanechnikov and Gaussian kernels), $K(x)$, is given by ,
$$
\hat{f}_K = \frac{1}{nh^d}\sum_{i=1}^{n}K(\frac{x-x_i}{h})
$$
where $h$ defines the radius of kernel. The radially symmetric kernel is defined as,
$$
K(x)=c_kk(\|x\|^2)
$$
where $c_k$ represents a normalization constant.

A kernel is a function that satisfies the following requirements:

1. $\int_{R^d}\phi(x)=1$
2. $\phi (x) \geq 1$

Some examples of kernels include:

1. Rectangulat
   $$
   \phi(x)=\left\{\begin{matrix}
   1 & a\leq x\leq b \\
   0 & else
   \end{matrix}\right.
   $$

2. Gaussian
   $$
   \phi(x)= e^{-\frac{x^2}{2\sigma^2}}
   $$

3. Epanechnikov
   $$
   \phi(x) =\left\{\begin{matrix}
   \frac{3}{4}(1-x^2) & |x| \leq 1 \\
   0 & else
   \end{matrix}\right.
   $$

Taking the derivative of:
$$
\nabla \hat{f}(x)=\frac{2c_{k,d}}{nh^{d+2}}\left[\sum_{i=1}^{n}g\left(\|\frac{x-x_i}{h}\|^2 \right) \right]\left[\frac{\sum_{i=1}^n x_ig(\|\frac{x-x_i}{h}\|^2)}{\sum_{i=1}^n g(\|\frac{x-x_i}{h}\|^2)}-x\right]
$$
where $g(x)=-k'(x)$ denotes the derivative of the selected kernel profile.

The mean shift procedure from a given point $x_t$ is:

1. Compute the mean shift vector $m$:
   $$
   \left[\frac{\sum_{i=1}^n x_ig(\|\frac{x-x_i}{h}\|^2)}{\sum_{i=1}^n g(\|\frac{x-x_i}{h}\|^2)}-x\right]
   $$

2. Translate the density window:
   $$
   x_i^{t+1}=x_i^t+m(x_i^t)
   $$

3. Iterate until convergence
   $$
   \nabla f(x_i) = 0
   $$

