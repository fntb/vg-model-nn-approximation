
## Notes on the Loss

Recall that when estimating the expectation of some $X$, written $m$, the TCL tells us that the empirical mean $\hat{m}_p = \frac{1}{p} \sum_{j=1}^p X_j$ converges in law : $\sqrt{p}(\hat{m}_p - m) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$. 

If the variance is unknown, we can estimate it using the unbiased variance estimator $\hat{\sigma}^2 = \frac{1}{p-1} \sum_{j=1}^p (X_j - \hat{m}_p)^2$, which, by the LGN and Slutsky's theorem, converges too : $\sqrt{p}\frac{(\hat{m}_p - m)}{\hat{\sigma}} \xrightarrow{d} \mathcal{N}(0, 1)$.

Consider then having $\{X_j(\theta_i)\}$ where the law of $X$ depends on some parameter $\theta$ and write $m(\theta) = \mathbb{E}[X(\theta)]$. 

With $p$ large, we can reasonably make the assumption $\hat{m} (\theta) \sim \mathcal{N}(m(\theta), \frac{\hat{\sigma}^2 (\theta)}{p})$.

Thus for $i \in \{1 \dots n\}$, the MLE is obtained by minimizing the negative log likelihood :

$$l(\theta \vert \{\hat{m}(\theta), \hat{\sigma}(\theta)\}) = \sum_{i=1}^n \frac{(\hat{m}(\theta_i) - m(\theta_i))^2}{\hat{\sigma}^2(\theta_i) / p}$$


Now to put this in context, we have a stochastic function $f$ and a dataset of parameters $\set{\theta_i}$, we want to learn $\theta \to \mathbb{E}[f(\theta)] =: m(\theta)$. To accomplish this we generate for each $\theta_i$, $p$ simulations $f(\theta_i)_j =: X_j(\theta_i)$ of which we take the empirical mean $\hat{m}(\theta_i)$ (that's just monte-carlo simulation) and unbiased variance estimator $\hat{\sigma}(\theta_i)^2$. We then train our neural network to approximate $\theta \to m(\theta)$ on the aforementionned criteria, which justifies this choice of weighted MSE.

There remains two potential issues : 

- First, since $p$ is going to be large, $\sigma_i$ may be low ;
- Second, we are doing option pricing here, do we really need to care about predictions being less than some distance $\epsilon$ to the true price ? Intuition says less than 0.01 (a cent) is unecessary, quick research says than it is for sensitivity analysis, ghost arbitrage, and batched bets (many contracts for a single prediction, which multiplies the error). The industry standard seems to be $10^{-6}$ but since we work with `float32` we'll stick to $10^{-4}$. 
