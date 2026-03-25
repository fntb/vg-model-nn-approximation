
## Notes on the Loss

#### Justifying the Weighted MSE

Recall that provided we have observations $\set{X_j}_{j \in \set{1 \dots p}}$ we can estimate the expectation $m := \mathbb{E}[X]$ with the empirical mean $\hat{m} = \frac{1}{p} \sum_{j=1}^p X_j$. By the TCL this estimator converges in law : 

$$\sqrt{p}(\hat{m} - m) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$ 

If the variance is unknown, we can estimate it using the (debiased) empirical variance $\hat{\sigma}^2 = \frac{1}{p-1} \sum_{j=1}^p (X_j - \hat{m})^2$, which - by the LGN, Slutsky's theorem, and the previous result - also converges in law : 

$$\sqrt{p}\frac{(\hat{m} - m)}{\hat{\sigma}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Consider then having observations $\set{X_j(\theta_i)}_{j \in \set{1 \dots p}, i \in \set{i \dots q}}$ where the law of $X$ depends on some parameter $\theta$ modeled as a random variable. 

Let us write $m(\theta) = \mathbb{E}[X(\theta)]$. **For $p$ large enough**, the previous result justifies the assumption


$$\sqrt{p}\frac{(\hat{m}(\theta) - m(\theta))}{\hat{\sigma}(\theta)} \sim \mathcal{N}(0, 1)$$

Thus, **conditionned on $\set{\hat{m}(\theta), \hat{\sigma}(\theta)}$**, $m(\theta)$ can be supposed to follow a gaussian distribution with mean $\hat{m}(\theta)$ and variance $\frac{\hat{\sigma}(\theta)^2}{p}$. We can then express the conditional negative log likelyhood (without constant terms) as :

$$l(\theta \vert \hat{m}(\theta), \hat{\sigma}(\theta)) = \sum_{i=1}^q \frac{(\hat{m}(\theta_i) - m(\theta_i))^2}{\hat{\sigma}^2(\theta_i) / p}$$


Now to put this in context, we have a stochastic function $f$ and a dataset of parameters $\set{\theta_i}$, we want to learn a parametric function $m_\phi$ (i.e. a neural network) to approximate $m : \theta \to \mathbb{E}[f(\theta)]$. 

To accomplish this we generate, for each $\theta_i$, $p$ simulations $f(\theta_i)_j$ (the $X_j(\theta_i)$ in our previous explanation) of which we take the empirical mean $\hat{m}(\theta_i)$ (i.e. the monte-carlo simulation of $\mathbb{E}[f(\theta_i)]$) and empirical variance $\hat{\sigma}(\theta_i)^2$. 

We can then train our neural network by gradient descent to minimize the aforementionned negative log likelihood (which is, in the end, a simple weighted MSE) :

$$\operatorname{WMSE}(m_\phi(\theta_i), \hat{m}(\theta_i), \hat{\sigma}^2(\theta_i) / p) = \sum_{i=1}^q \frac{(\hat{m}(\theta_i) - m_\phi(\theta_i))^2}{\hat{\sigma}^2(\theta_i) / p}$$


#### Conclusion and Going Further

- The weighted MSE is not a default choice, but instead is justified as a proper Maximum Likelihood Estimation ;
- Our only two hypothesis where "$p$ is large" - which it is at $2^7$ - and "conditionned on $\set{\hat{m}(\theta), \hat{\sigma}(\theta)}$" - which translates to "Assuming the MC estimate is the true parameter".

- Since $p$ is indeed large, $\frac{\hat{\sigma_i}^2}{p}$ may be very low, which could at best explode our loss and thus our gradient at worse cause `NaNs` ; _We ough to similarly derive a proper loss to account for this "machine precision noise"_
- Recall that we are modeling option pricing, do we really need to care about prediction precision under some $\epsilon$ ? Naively a precision under $10^{-2}$ (a cent) seem unecessary, however it is for sensitivity analysis, ghost arbitrage, and batched bets (many contracts depending on a single prediction, which multiplies the error by the number of contracts). The industry standard seems to be $10^{-6}$. Since we work with `float32` we'll stick to $10^{-4}$ in our evaluation metrics. 
