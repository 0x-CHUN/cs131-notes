# Edge

## Edge detection

Edge detection ： Indentify sudden changes in an image.

Edges :

* Extract information，recognition objects
* Recover geometry and viewpoint

Origins of edges :

* Surface normal discontinuity
* Depth discontinuity
* Surface color discontinuity
* illumination discontinuity

## Image Gradients

Derivatives : 
$$
\frac{df}{dx} = \lim_{\Delta x\to 0}\frac{f(x)-f(x-\Delta x)}{\Delta x}=f'(x)=f_x
$$
Discrete derivate in 2D :
$$
\nabla f(x, y)=\left[\begin{array}{l}
\frac{\partial f(x, y)}{\partial x} \\
\frac{\partial f(x, y)}{\partial y}
\end{array}\right]=\left[\begin{array}{l}
f_{x} \\
f_{y}
\end{array}\right]
$$

## A simple edge detector

An edge is a place of rapid change in the image intensity function

The gradient of an image : 
$$
\nabla f=[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]
$$
The gradient direction is given by $\theta = tan^{-1}(\frac{\partial f}{\partial y}/ \frac{\partial f}{\partial x})$

The edge strength is given by the gradient magnitude : $\|\nabla f\|=\sqrt{(\frac{\partial f}{\partial x})^2 + (\frac{\partial f}{\partial y})^2}$

Effects of noise:

* Finite difference filters respond strongly to noise
* Solution : Smoothing the image

Smoothing with different filters:

* Mean smoothing
* Gaussian

 Criteria for an “optimal” edge detector:

* Good detection: the optimal detector must minimize the probability of false positives (detecting spurious edges caused by noise), as well as that of false negatives (missing real edges)
* Good localization: the edges detected must be as close as possible to the true edges
* Single response: the detector must return one point only for each true edge point; that is, minimize the number of local maxima around the true edge

## Sobel Edge detector

Sobel Operator (= Smoothing + differentiation):
$$
G_x = \begin{bmatrix}
1  & 0 & -1 \\
2  & 0 & -2\\
1  & 0 & -1
\end{bmatrix} \
G_y = \begin{bmatrix}
1  & 2 & 1 \\
0  & 0 & 0\\
-1  & -2 & -1
\end{bmatrix}
$$
Magnitude: $G = \sqrt{G_x^2 + G_y^2}$

Angle or direction of the gradient : $\theta = arctan(\frac{G_y}{G_x})$

Sobel Filter Problems :

* Poor Localization
* Thresholding value favors certain directions over others

## Canny edge detector

* Suppress Noise

* Compute gradient magnitude and direction

* Apply Non-Maximum Suppression
  
  * $$
    M(x,y) = \left\{\begin{matrix}
    |\nabla G|(x,y)  &  |\nabla G|(x,y)>|\nabla G|(x',y') \& |\nabla G|(x,y)>|\nabla G|(x'',y'')\\
     0 & other
    \end{matrix}\right.
    $$
  
* Use hysteresis and connectivity analysis to detect edges

  * Avoid streaking near threshold value
  * Define two thresholds: Low and High
    * If less than Low, not an edge
    * If greater than High, strong edge
    * If between Low and High, weak edge

## Hough transform

Hough transform can detect lines, circles and other structures **ONLY** if their parametric equation is known

Line : $y_i = a*x_i + b \to b = -a*x_i + y_i$ 

So a single point in $x_1,y_1$-space gives a line in $(a,b)$ space.

Algorithm for Hough transform : 

* Quantize the parameter space (a b) by dividing it into cells
* This quantized space is often referred to as the accumulator cells.
* Count the number of times a line intersects a given cell.
  * For each pair of points $(x_1, y_1)$ and $(x_2, y_2)$ detected as an edge, find the intersection $(a’,b’)$ in $(a, b)$ space.
  * Increase the value of a cell in the range $[[a_{min}, a_{max}],[b_{min},b_{max}]]$ that $(a’, b’)$ belongs to.
  * Cells receiving more than a certain number of counts are assumed to correspond to lines in $(x,y)$ space.

Advantages : 

* Conceptually simple.
* Easy implementation
* Handles missing and occluded data very gracefully.
* Can be adapted to many types of forms, not just lines

Disadvantages : 

* Computationally complex for objects with many parameters.
* Looks for only one single type of object
* Can be “fooled” by “apparent lines”.
* The length and the position of a line segment cannot be determined.
* Co-linear line segments cannot be separated.