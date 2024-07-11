# Experiment write-ups

This section should be helfpul to understand where we are at. Let's order the write-ups by how promising or complete they are.


## Routing
I call a "router" a model that picks which of our classifiers we should use to classify a given $(64,161)$ EEG recording. I think routing may be quite powerful because of this analysis (in `ensemble.py`): consider the set of subjects correctly classified by each of our best models, and see how they differ.

If you sum the number of successfully classified rows in the 4-model venn diagram below you get a theoretical upper bound of 88%. Adding more models increases this bound.

Of course, we don't know that building routers will actually be easier than building classidiers, but it seems like a promising idea. 

I spent a lot of time trying to build CNN-based routers with no success, however an FgMDM based router did work. Below is a plot of a few router architectures, and their accuracy (I don't remember if this was for the 3-model routing problem or the 4-model routing problem, I think it was 3-models).


<p align="center">
  <img src="https://github.com/trialan/eeg/assets/16582240/cc6db827-2072-458b-8e97-e0d6b1a0dfdb" alt="overnight_run" width="45%" height="300px">
  <img src="https://github.com/trialan/eeg/assets/16582240/2abec77f-cad7-4e7d-b53c-6cceefca6fc8" alt="Their version" width="45%" height="300px">
</p>

As we can see in the results on `router.py` below: the score of the meta-classifier using an FgMDM-based router with 24 eigen-components is an improvement on the Xu et al. top score of 64.2%.


```python
###### CSP+LDA Router score: 0.4166666666666667

###### Meta-clf score (CSP+LDA router): 0.6340782122905028

###### EDFgMDM Router score: 0.4041666666666667

###### Meta-clf score (EDFgMDM router): 0.6634078212290503
```

_Current question marks:_
- Why does the router with the best accuracy, namely CSP + LDA, not get the best final score? This makes no sense.
- Does this result still hold when you do proper 5-fold CV? For ease of implementation, `router.py` does a single fold. This was reasonable to implement the idea, but to be bullet proof it needs to be 5-fold so that we're comparing apples to apples.


## Picking specific channels
Perhaps (as in the fNIRS literature says Nyx) it would help to only use a subset of the channels. So I ran some linear regressions to pick the best channels, and then I train FgMDM models where I only give it the top N channels, N ranges from 1 to 64. This beats vanilla FgMDM by 1.8% with a score of 0.637945 (vs 0.6199 for vanilla FgMDM). Results are plotted below.

![FgMDM (N best channels)](https://github.com/trialan/eeg/assets/16582240/488d5f50-5864-4ea8-974c-dbc4baa87825)

Perhaps we could now do this (top 24 channels) and then do Laplacian + FgMDM (24 eigenmodes) on this (the current best "pure" (non-router based) algo). If that beats Laplacian+FgMDM (24 eigenmodes), then we can put it in the router to have a "best in class" attempt/model. Let's see. I will leave the code in `channel.py` un-touched now, and use another file.

Just to be sure I don't lose the sorting of channels this is the sorting I used in this experiment.

```python
array([63, 62, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
       14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 30, 31, 32,
       48, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 47, 33, 46,
       45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34,  0])
```

_Current question marks_:
- Why can't I reproduce this sorted list of channels with the code right now?
- How do we use this result to improve our overall score? Just putting it as a model in the router doesn't seem to help. Is that right? Can we combine this idea with the Laplacian + FgMDM model of Xu et al. to beat their performance without a router, and then put _that_ in the router?



## Ordering the eigenvectors

When we look at the plot of performance vs the number of eigenvectors, it seems like some eigenvectors help, and others hurt the score. We haven't yet got a clever way of ordering eigenvectors, but the experiment below, where I do Laplacian + FgMDM with N eigenmodes, but the order of eigenvectors is shuggled, suggests that a clever way of sorting them may yield performance improvements.

One complication with this idea is that some modes help only when in combinaton with othe.

![random_shuffle_eigenvec](https://github.com/trialan/eeg/assets/16582240/8c4d93e5-bcc3-449e-8fc2-4d0ae5f92838)

## Fourier transforming coefficient matrix

![10subjects_3rdEigenmode_AverageOfFourierTransform](https://github.com/trialan/eeg/assets/123100675/d06ca0df-3b80-45f5-b45b-e6acbc8895c9)
The Fourier transforms of the coefficient of the third eigenmode as a function of time over each epoch in category '0' (probably 'hands') and category '1' for 10 subjects was taken. Then those fourier transforms were averaged. We see two consistent features, a dip in power around 0.1 for '1' and a difference in slope around the tails.
Here are the fourier transforms of eigenmode decomposition coefficients for the first 20 eigenmodes, orange is for hands, blue is for feet (or vice versa). The FTs for all 'hands' epochs were averaged (for all subjects) and same for feet. Notice eigenmode 16 - it might be used for distinguishing between the two?
![fourier_galore_allmodes](https://github.com/trialan/eeg/assets/123100675/d37d97f0-bc04-4206-b7e0-ddeb80e4031c)

## Upscaling mesh
![150centr](https://github.com/trialan/eeg/assets/123100675/f4524adf-fe6a-4d13-8a89-f462477e968d)
Produced by resampling the electrode psoitions so that they remain in the convex hull and new triangles are created.

## Laplacian Spatial Patterns dimensionality reduction
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

## Ensembling
Write up in ensembling.py, tl;dr this wasn't particularly helpful.


## CNNs with time series
In this experiment I trained CNNs to learn the time series. Why? My logic was: the one trick that sort of worked has been this jittering. Deep learning models are great, but need lots of data. But we don't have that much data. Ok. With jittering we can get infinite amounts of data. So let's start with that. This is the default results (no data augmentation):

![CNN_raw_100pct](https://github.com/trialan/eeg/assets/16582240/17923583-dee7-4d85-b459-f7b023d4c64d)

The best validation loss here is `0.6928` (note this is averaged over 5 folds, still pretty ugly val losses!). Using Andrew Ng's [nuts and bolts of deep learning](https://www.datasciencecentral.com/nuts-and-bolts-of-building-deep-learning-applications-ng-nips2016/) we see that our 'dev error' (I call this validation loss) is high, so we should add more data. Before we do this, let's do a quick abalation experiment to verify that the validation loss is indeed worse (higher) for less data.

Over the five folds the best validation score can be found by taking the min of the validation losses for that fold:

```python
In [1]: np.mean([np.min(v) for v in split_val_losses])
Out[1]: 0.6928343726404058

In [2]: np.std([np.min(v) for v in split_val_losses])/np.sqrt(5)
Out[2]: 0.00010629982104669662
```

So actually our performance with the raw CNN on the full time-series is $0.6929 \pm 0.0001$. I'll report on what model accuracy this corresponds to later. 

To do this ablation experiment, I re-run the above experiment with 50% less data in the train set (but the same amount of data in the validation set).

```python
    for train_index, val_index in tqdm(cv.split(X), desc="Split learning"):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        X_train = get_fraction(X_train, 0.5)
        y_train = get_fraction(y_train, 0.5)
        train_loader, val_loader = get_dataloaders(X_train, X_val,
                                                   y_train, y_val)
```

And this gave the result $0.6929 \pm 0.0001$ (which I think was about 50% accuracy, i.e. garbage). The magnitude on this is a bit hard to interpret but smaller numbers are better, and more data had a smaller number, so the Andrew Ng flowchart gave us the right idea and we will now experiment with data augmentation. Let's 3x the data using jittering. Now, in `jitter_augment.py` we augment the number of electrodes. Here we have evidence for augmenting the number of samples, so let's do that instead. I use the `augment_data` function in `cnn.py`. This is what I get:

![3x_jitter_CNN](https://github.com/trialan/eeg/assets/16582240/fd89b2ff-3a18-4739-addc-0a555080dbe7)

```python
In [1]: np.mean([np.min(v) for v in split_val_losses])
Out[1]: 0.6930747129882157

In [2]: np.std([np.min(v) for v in split_val_losses])/np.sqrt(5)
Out[2]: 3.35874317722525e-05
```

--> This didn't help. That's a bit odd, needs further reflecting.

## Data-augmentation / reduction experiments
Write up as a comment in the code file. Taken's theorem from [this paper](https://arxiv.org/pdf/2403.05645?fbclid=IwZXh0bgNhZW0CMTAAAR1wcNdM6sIvx3LgeoNmmbgoFQp5Tr9sF7Ud651u5KMlQf6zNsX0VNQynHU_aem_rkkPO4cvOQQCELS2vtudVQ)
,which does something a bit like FgMDM but more fancy geometric deep learning, is quite interesting. But my take away from the experiment at data_augmentation_experiment.py is that
this only helped them because they use 3 electrodes, and is unlikely to help us.

## Using SVC instead of LDA
CSP + SVC slightly under-performed CSP+LDA. I attribute this to SVC needing more data, but haven't dug too deep as this seems like a boring idea.

## Investigating these "Common Spatial Patterns"
This comes up a lot in the papers results. Seems like a sort of PCA-but-not-PCA thing for "rotating" the data into a nicer shape for the classifier. Their best two results are
Laplacian + FgMDM and PCA + FgMDM, i.e. dimensionality reduction trick + FgMDM. Why not CSP + FgMDM? I plotted it below, similar peak performance at 18 components. They probably didn't plot it because the curve looks bad.

![CSP+FgMDM](https://github.com/trialan/eeg/assets/16582240/2cf0725b-d9cf-46b9-83c9-5685f8f10641)

The CSP does this thing where it (1) finds the vectors on which to project the data, then it drops the time dimension by taking avg power (avg square of each element), and then applying some sort of scaling (either log scaling or z-scaling).

