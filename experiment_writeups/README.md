# Experiment write-up

## Brain-geometry informed experiments

### Laplacian Spatial Patterns dimensionality reduction
In this experiment I want to: 

1. Re-write the (n_channels, n_times) sub-matrices in the eigenbasis.
2. Do the CSP trick of taking average power + scaling to drop the time dimension.
3. Apply a standard classifier like LDA.

This did very poorly without the scaling (peak score about 54%, which is almost as bad as random guessing). With the scaling it also has a bad score but has a very interesting shape

![LSP+LDA (64)](https://github.com/trialan/eeg/assets/16582240/2269fc8e-9d92-4c52-b2f3-2b79d66caee4)

This was using finite element methods for computing the eigenvectors, `sphara_basis_unit = sb.SpharaBasis(mesh, 'fem')`. However when I use `inv_euclidean` we do not get this linear trend, but instead:

![inv_euclidean_LSP+LDA](https://github.com/trialan/eeg/assets/16582240/82b68daf-c039-4721-9cd8-2b7852f4cff4)

Which is not promising, I would have hoped that the linear trend worked in both cases.

The linear trend suggests: use more eigenvectors and more electrodes, and you'll get a better score. So we do the simplest thing: get "synthetic electrodes" by jittering the data, and use spharapy's mesh to generate more eigenvectors. This got the score up to 59.7% (up from 56% with the first CSP+LDA experiment).

```python
eigenvectors = get_256D_eigenvectors()

def jitter(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=np.mean(x), scale=sigma, size=x.shape)


def augment_subX(subX):
    assert subX.shape == (64, 161)
    new_subX = list(subX)
    for j in range(256 - 64):
        ix = j%64
        se = np.std(subX[ix]) / np.sqrt(161)
        new_electrode = jitter(subX[ix], sigma=se/300)
        new_subX.append(new_electrode)
    new_subX = np.array(new_subX)
    assert new_subX.shape == (256, 161)
    return new_subX


def transform_data(X):
    X_prime = np.array([augment_subX(subX) for subX in tqdm(X)])
    return X_prime

```



### Ordering the eigenvectors?
This experiment below suhggests we may want a clever way of sorting the order in which we add eigenvectors to the reduced-dimensions. This was inspired by AK's comment on
the fact that some eigenmodes help the score, and others hurt the score. So you'd want to only keep those modes which help the score. However some modes help only when
in combinaton with other.

![random_shuffle_eigenvec](https://github.com/trialan/eeg/assets/16582240/8c4d93e5-bcc3-449e-8fc2-4d0ae5f92838)


## ML experiments
### Fourier transforming coefficient matrix
![10subjects_3rdEigenmode_AverageOfFourierTransform](https://github.com/trialan/eeg/assets/123100675/d06ca0df-3b80-45f5-b45b-e6acbc8895c9)
The Fourier transforms of the coefficient of the third eigenmode as a function of time over each epoch in category '0' (probably 'hands') and category '1' for 10 subjects was taken. Then those fourier transforms were averaged. We see two consistent features, a dip in power around 0.1 for '1' and a difference in slope around the tails.
Here are the fourier transforms of eigenmode decomposition coefficients for the first 20 eigenmodes, orange is for hands, blue is for feet (or vice versa). The FTs for all 'hands' epochs were averaged (for all subjects) and same for feet. Notice eigenmode 16 - it might be used for distinguishing between the two?
![fourier_galore_allmodes](https://github.com/trialan/eeg/assets/123100675/d37d97f0-bc04-4206-b7e0-ddeb80e4031c)


### TODO: ensembling!
This must help if the approaches are sufficiently orthogonal.

### Data-augmentation / reduction experiments
Write up as a comment in the code file. Taken's theorem from [this paper](https://arxiv.org/pdf/2403.05645?fbclid=IwZXh0bgNhZW0CMTAAAR1wcNdM6sIvx3LgeoNmmbgoFQp5Tr9sF7Ud651u5KMlQf6zNsX0VNQynHU_aem_rkkPO4cvOQQCELS2vtudVQ)
,which does something a bit like FgMDM but more fancy geometric deep learning, is quite interesting. But my take away from the experiment at data_augmentation_experiment.py is that
this only helped them because they use 3 electrodes, and is unlikely to help us.

### Using SVC instead of LDA
CSP + SVC slightly under-performed CSP+LDA. I attribute this to SVC needing more data, but haven't dug too deep as this seems like a boring idea.

### Investigating these "Common Spatial Patterns"
This comes up a lot in the papers results. Seems like a sort of PCA-but-not-PCA thing for "rotating" the data into a nicer shape for the classifier. Their best two results are
Laplacian + FgMDM and PCA + FgMDM, i.e. dimensionality reduction trick + FgMDM. Why not CSP + FgMDM? I plotted it below, similar peak performance at 18 components. They probably didn't
plot it because the curve looks bad.

![CSP+FgMDM](https://github.com/trialan/eeg/assets/16582240/2cf0725b-d9cf-46b9-83c9-5685f8f10641)

The CSP does this thing where it (1) finds the vectors on which to project the data, then it drops the time dimension by taking avg power (avg square of each element), and then
applying some sort of scaling (either log scaling or z-scaling).

