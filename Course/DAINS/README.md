# Data and Artifical Intelligence for Numerical Simulations

<p align="justify">
Computer simulation (FEA) can model metal bending and stress very well, but running them over and over takes too much time. Through this lecture, projects were conducted combining simulations, active learning, and deep learning to speed up the process while keeping accuracy.
</p>

## Project 1: Physics
<p align="justify">
Load-controlled bending simulation was performed on a square apecimen utilized a bilinear elastoplastic model with kinematic hardening. Equilibrium was resolved via a modified "Initial Stiffness" Newton-Raphson scheme (Eq. 1) [1] paired with radial return mapping for von Mises stress integration (Eq. 2) [2].

$$\mathbf{K}_t \Delta u = \mathbf{f}_{t+\Delta t}^{ext} - \mathbf{f}_t \quad (1)$$
$$\mathbf{s}_{n+1}^T = \mathbf{s}_n + 2G \Delta \mathbf{e}_{n+1}, \quad \mathbf{s}_{n+1} \equiv P(\mathbf{s}_{n+1}^T) \quad (2)$$

Pseudo-time steps are solved iteratively until residual tolerances or maximum iteration limits are met, upgrading internal stresses and plastic history variables at each step.
</p>

## Project 2: Optimization
<p align="justify">
Using the Project 1 Finite Elemenet model as a black box, multi-objective optimization maximized maximum von Mises stress and minimized maximum final displacement across varied material stiffness (Young's modulus) and varied prescribed boundary displacement. Executed over 20 Ax active learning trials, space-filling Sobol sampling preceded Gaussian Process modeling (3). The GP prior and dataset induce a posterior f(x), while acquisition function (4) determines the next evaluation point in X via proxy optimization (5) [3].

$$ \\{x_n, y_n\\}_{n=1}^N, \text{ where } y_n \sim \mathcal{N}(\mu, \sigma^2) \quad (2) $$
$$a : \mathcal{X} \rightarrow \mathbb{R}^+ \quad (4)$$
$$ x_{\text{next}} = \arg\max_x a(x) \quad (5) $$
</p>

## Project 3: Predicition
<p align="justify">
Two surrogate models – a DNN and a 2D CNN – were trained on preprocessed (shuffled, split, normalized) elastoplastic FEA data from Projects 1 and 2. The DNN maps scalar inputs maximum displacement, Young's modulus, and yield stress to scalar outputs maximum force and maximum von Mises stress. Meanwhile, the CNN predicts the full von Mises stress field given boundary displacement, Young's modulus, and yield stress.
</p>

<p align="justify">
Implemented in TensorFlow/Keras, both models use time-distributed surrogate models weight sharing across time steps and were trained with Adam (α = 10-3), MSE loss, and MAE metrics. The sequence-based DNN contains three hidden layers (64, 64, 32 ReLU units) with a linear output layer, trained for 300 epochs (batch size 2, 25% validation split). The 2D CNN features four 32-filter (7×7) layers and one 16-filter (5×5) layer with ReLU and same padding, ending in a 1×1 convolutional layer. It was trained for 200 epochs (batch size 8, 20% validation split).
</p>

## Reference

1. Owen, D.R.J., Gomez, C.M.B.: An appraisal of numerical solution techniques for elasto-plastic and elasto-viscoplastic material problems. Convergence 1(1), (1981).
2. Simo, J.C., Taylor, R.L.: Consistent tangent operators for rate-independent elastoplasticity. Computer Methods in Applied Mechanics and Engineering 48(1), 101-118 (1985).
3. Snoek, J., Larochelle, H., Adams, R.P.: Practical Bayesian optimization of machine learning algorithms. Advance in Neural Information Processing Systems 25, 2960-2968 (2012).




