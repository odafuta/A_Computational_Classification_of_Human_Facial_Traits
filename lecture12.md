# LECTURE 02: Machine Learning I: Conventional Machine Learning

**Dr. Suyong Eum**  
**OSAKA UNIVERSITY**

---

## 1. Lecture Outline

1. **Principal Component Analysis (PCA)**
   - Feature selections
   - Dimension reduction

2. **Support Vector Machine (SVM)**
   - Hard margin SVM: linear classification
   - Kernel trick: nonlinear classification

---

## 2. Principal Component Analysis (PCA)

### Principal Component Analysis (PCA): Definition

A statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

*In Wikipedia:*

$$\begin{bmatrix} 8 & 1 \\ 1 & 8 \end{bmatrix} \rightarrow \begin{bmatrix} 10 & 0 \\ 0 & 1 \end{bmatrix}$$

### Principal Component Analysis (PCA): Intuition

**How to select principal components?**

One that captures the largest variance of the data points.

Intuitively speaking, you can observe more data from the direction ① than any other direction, and then from the direction ②, you can observe the data with the least redundancy compared to the direction ①.

**How to find the principal components showing the largest variance?**

1. Find the covariance matrix of data points.
2. Obtain the eigen values and vectors of the covariance matrix: eigen value decomposition.
3. Sort the eigen vectors in descending order in terms of their corresponding eigen values.
4. An eigen vector with the largest eigen value becomes the first principal component.

### Example

Given data points X:

$$X = \begin{bmatrix} -2 & -1 & 1 & -1 & 1 & 2 \\ -2 & -1 & -1 & 1 & 1 & 2 \end{bmatrix}$$

The covariance matrix is:

$$\text{cov}(X) = \begin{bmatrix} 2.4 & 1.6 \\ 1.6 & 2.4 \end{bmatrix}$$

Eigen value decomposition is $\text{cov}(X) = V\Lambda V^T$.

From `[vec, val] = eig(cov(x))`:

$$\text{vec} = \begin{bmatrix} -0.70711 & 0.70711 \\ 0.70711 & 0.70711 \end{bmatrix}$$

$$\text{val} = \begin{bmatrix} 0.8 & 0 \\ 0 & 4.0 \end{bmatrix}$$

The columns of `vec` are the eigenvectors $(v_2, v_1)$, and `val` is a diagonal matrix of eigenvalues.

### Singular Value Decomposition (SVD)

Actually, there is a more convenient way of doing it, which is called **"Singular Value Decomposition"** or **SVD**.

$$X = U\Sigma V^T$$

From this, we can derive the relationship with eigen decomposition:

$$X^TX = (U\Sigma V^T)^T(U\Sigma V^T) = V\Sigma^T U^T U\Sigma V^T = V\Sigma^2 V^T$$

Therefore, $\Lambda = \Sigma^2$.

### Now we know how to find the principal components

**Principal Component Analysis (PCA)** is nothing but finding principal components of a given data set.

- Principal components are the directions where you look at the data set, which provides the most information of the data set.
- They're equivalent to eigen vectors which can be found by SVD or EVD.
- The eigen value corresponding to each eigen vector represents how widely the data set is spread along the direction which is perpendicular to the eigen vector.

### Dimension Reduction

- A data point is defined by several, let's say, features.
- The number of features to define a data point is called the **dimension** of the data.
- High dimension data implies that it contains much information. Sometimes, we reduce its dimension, e.g., to visualize the data or to efficiently analyze them.
- **PCA can reduce the dimension without losing relatively less information** of the data.
- To reduce the dimension, you project the data points onto the eigenvector space. For example, to reduce from 2D to 1D, project the data onto the first principal component $(v_1)$.

### Dimension Reduction: An Example

Let's say, we have one image representing one data point (e.g., an 8×8 image of a handwritten digit). We can represent the data by all pixels, which are 64 in this case. In other words, it is **64-dimensional data**.

What happens if we reduce its dimension to **2 dimensions** using PCA? The 64-dimensional data points can be plotted in a 2D space (1st Principal Component vs. 2nd Principal Component). This allows for:
- **Data visualization**
- May even allow for **classification** by drawing a line to separate different classes of data.

---

## 3. Support Vector Machine (SVM)

### Why Support Vector Machine?

- **Most widely used classification approach** (practical)
  - Linearly separable data set
  - Non-linearly separable data set
- **Supported by well defined mathematical theories**
  - Geometry
  - Optimization

### Terminology used in SVM

- **Decision boundary (Hyperplane)**: The line that separates the classes.
- **Support Vectors**: The data points that are closest to the decision boundary.
- **Support lines**: The lines passing through the support vectors, parallel to the decision boundary.
- **Margin**: The distance between the support lines. SVM aims to maximize this margin.

### Problem Formulation

The goal is to find a decision boundary that **maximizes the margin**. This can be formulated as an optimization problem.

The distance (margin) $||r||$ is $\frac{1}{||w||}$. So, maximizing the margin is equivalent to minimizing $||w||$.

The problem can be written as:

$$\min \frac{1}{2}||w||^2$$

subject to:
$$t_n(w^Tx_n + w_0) \geq 1, \forall n$$

where $t_n$ is the class label (+1 or -1). This is a **Quadratic Programming problem**.

### How about non-linearly separable case?

The above formulation works for linearly separable data. For non-linear data, we use the **Kernel Trick**.

---

## 4. Kernel Trick

### Lagrange method for an optimization problem with inequality constraints

To solve the constrained optimization problem of SVM, we use the method of **Lagrange multipliers**. For a problem like $\min x^2$ subject to $x \geq b$, the Lagrangian is $L(x,\lambda) = x^2 - \lambda(x-b)$ with $\lambda \geq 0$.

A key result is the **complementary slackness condition**: $\lambda(x-b) = 0$.

### Convert the quadratic problem in SVM to Lagrange optimization problem

The **primal problem** is:

$$\min_{w,w_0} \max_{\lambda} L(w,w_0,\lambda) = \frac{1}{2}w^Tw - \sum_{n=1}^{N} \lambda_n(t_n(w^Tx_n + w_0) - 1)$$

subject to $\lambda_n \geq 0$.

We can switch the min and max to get the **dual problem**:

$$\max_{\lambda} \min_{w,w_0} L(w,w_0,\lambda)$$

subject to $\lambda_n \geq 0$.

This is valid under the **Karush-Kuhn-Tucker (KKT) conditions**.

### Dual problem of the quadratic problem

By taking derivatives of $L$ with respect to $w$ and $w_0$ and setting them to zero (stationarity condition), we get:

$$w = \sum_{n=1}^{N} \lambda_n t_n x_n$$

$$\sum_{n=1}^{N} \lambda_n t_n = 0$$

Substituting these back into the Lagrangian, we get the dual problem which only depends on $\lambda$:

$$\max_{\lambda} L(\lambda) = \sum_{n=1}^{N} \lambda_n - \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} t_n t_m \lambda_n \lambda_m x_n^T x_m$$

subject to $\lambda_n \geq 0$ and $\sum_{n=1}^{N} \lambda_n t_n = 0$.

This is again a **quadratic programming problem**, but now in terms of $\lambda$.

### Let's summarize

1. We solve the dual problem to find the Lagrange multipliers $\lambda_n$.
2. Many of the $\lambda_n$ will be zero. The data points $x_n$ for which $\lambda_n > 0$ are the **support vectors**.
3. From the complementary slackness condition $\lambda_n(t_n(w^Tx_n + w_0) - 1) = 0$, for support vectors, we have $t_n(w^Tx_n + w_0) = 1$.
4. We can find $w$ using $w = \sum \lambda_n t_n x_n$ (sum over support vectors).
5. We can find $w_0$ using $w_0 = t_n - w^Tx_n$ for any support vector $x_n$.

### Kernel trick

If data is not linearly separable, we can map it to a **higher-dimensional space** where it becomes linearly separable. Let this mapping be $\phi(x)$.

The dual problem objective function depends on the inner product $x_n^T x_m$. In the new space, this becomes $\phi(x_n)^T \phi(x_m)$.

The **Kernel function** is defined as:
$$K(x_n, x_m) = \phi(x_n)^T \phi(x_m)$$

The trick is that we can compute $K(x_n, x_m)$ directly from $x_n$ and $x_m$ **without explicitly performing the mapping** $\phi$.

The dual problem becomes:

$$\min_{\lambda} \frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} t_n t_m \lambda_n \lambda_m K(x_n, x_m) - \sum_{n=1}^{N} \lambda_n$$

subject to $\lambda_n \geq 0$ and $\sum_{n=1}^{N} \lambda_n t_n = 0$.

### Example: Polynomial kernel of degree 2

$$K(x,y) = (x^Ty)^2$$

This kernel corresponds to a mapping $\phi$ to a higher-dimensional space. We can solve the SVM problem in this new space without ever explicitly calculating the coordinates in that space.

---

## 5. Summary

**PCA and SVM** are probably the most representative conventional machine learning algorithms.

### PCA
PCA helps you to manipulate a set of data in a way that:
- **Determining which features are important**
- **Reducing its dimension**, so that the data can be processed or visualized more efficiently

### SVM
SVM is a **classification method** founded on a well-defined mathematical framework, which can handle:
- **Linear classification problems**
- **Nonlinear classification problems**