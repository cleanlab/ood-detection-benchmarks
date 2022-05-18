"""
OOD detection via Mahalanobis distance and relative version thereof:
https://arxiv.org/pdf/2106.09022.pdf
"A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection"
"""

import numpy as np

## Original Mahalanobis
def fit_mahalanobis(features, labels, num_class):
    """ Apply this to the in-distribution training data.
        features = 2D numpy array
        labels = 1D numpy array
        num_class = len(set(labels)) typically
    """
    class_means = np.zeros((num_class,features.shape[1]))
    for i in range(num_class):
        class_inds = labels == i
        class_means[i] = np.mean(features[labels == i], axis=0)
    covar = np.zeros((features.shape[1],features.shape[1]))
    for i in range(num_class):
        class_n = np.sum(labels == i)
        class_covar = np.cov(features[labels == i].T)
        # centered_class_examples = (features[labels == i] - class_means[i])
        covar += class_covar * (class_n-1)
    covar = covar / len(labels)
    inverse_covar = np.linalg.pinv(covar)
    fit_output = {
        "class_means": class_means,
        "inverse_covar": inverse_covar,
    }
    return fit_output

def score_mahalanobis(features, fit_output):
    """ Use this for test data, produces one score per row of features. 
        Lower score indicates more OOD.
    """
    inverse_covar = fit_output["inverse_covar"]
    class_means = fit_output["class_means"]
    num_class = len(class_means)
    dists = np.zeros((len(features), num_class))
    for i in range(num_class):
        first_product = np.dot(features - class_means[i], inverse_covar)
        dists[:,i] = np.einsum('ij,ij->i', first_product, features - class_means[i])
    scores = np.min(dists, axis=1)
    return -scores

## Relative Mahalanobis Distance (RMD)
def fit_rmd(features, labels, num_class):
    """ Apply this to the in-distribution training data.
    """
    fit_output = fit_mahalanobis(features, labels, num_class)
    overall_mean = np.mean(features, axis=0)
    overall_covar = np.cov(features.T, bias=True)
    fit_output["overall_mean"] = overall_mean
    fit_output["overall_invcovar"] = np.linalg.pinv(overall_covar)
    return fit_output

def score_rmd(features, fit_output):
    """ Use this for test data, produces one score per row of features. 
        Lower score indicates more OOD.
    """
    mahalanobis_scores = score_mahalanobis(features, fit_output)
    overall_mean = fit_output["overall_mean"]
    overall_invcovar = fit_output["overall_invcovar"]
    first_product = np.dot(features - overall_mean, overall_invcovar)
    background_scores = np.einsum('ij,ij->i', first_product, features - overall_mean)
    return mahalanobis_scores + background_scores


if __name__ == "__main__":
    ### Script to test it out:
    sampsize = 100
    num_class = 5
    dimension = 3
    noise_frac = 0.03
    seed = 353

    def label_func(X, numclass):
        x0 = X[:,0]
        y = np.zeros(len(x0))
        percs = np.linspace(0, 1, numclass+1)
        for i in range(numclass):
            lower = np.quantile(x0,percs[i])
            upper = np.quantile(x0,percs[i+1])
            y[(x0 > lower) & (x0 <= upper)] = i
        return y.astype(int)


    # Generate train/test data:
    np.random.seed(seed)
    X = np.random.rand(sampsize, dimension)

    # y = np.random.randint(low=0, high=numclass, size=sampsize)
    y = label_func(X, numclass=num_class)

    Xtest = np.random.rand(sampsize, dimension)
    Xtest[0] = np.ones((dimension,))*1000  # 0th point is outlier 


    # Run mahalanobis scoring:
    m = fit_mahalanobis(X, y, num_class)
    scores_m = score_mahalanobis(Xtest, m)
    print(scores_m)

    # Run RMD scoring:
    r = fit_rmd(X, y, num_class)
    scores_r = score_rmd(Xtest, r)
    print(scores_r)
