# Experiment write-ups

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


### Picking specific channels
Perhaps (as in the fNIRS literature says Nyx) it would help to only use a subset of the channels. So I ran some linear regressions to pick the best channels, and then I train FgMDM models where I only give it the top N channels, N ranges from 1 to 64. This beats vanilla FgMDM by 1.8% with a score of 0.637945 (vs 0.6199 for vanilla FgMDM). Results are plotted below.

![FgMDM (N best channels)](https://github.com/trialan/eeg/assets/16582240/488d5f50-5864-4ea8-974c-dbc4baa87825)

Perhaps we could now do this (top 24 channels) and then do Laplacian + FgMDM (24 eigenmodes) on this (the current best "pure" (non-router based) algo). If that beats Laplacian+FgMDM (24 eigenmodes), then we can put it in the router to have a "best in class" attempt/model. Let's see. I will leave the code in `channel.py` un-touched now, and use another file.

## ML experiments

### Ensembling
Write up in ensembling.py, tl;dr this wasn't particularly helpful.

### Routing
I call a "router" a model that picks which of our classifiers we should use to classify a given $(64,161)$ EEG recording. I think routing may be quite powerful because of this analysis (in `ensemble.py`): consider the set of subjects correctly classified by each of our best models, and see how they differ. In the diagram below we see that there are 62+141 subjects correctly classified by 10-component CSP+LDA that 24-component Laplacian+FgMDM (our most powerful model) is unable to correctly label. By summing all the numbers on the diagram we can get a theoretical upper bound (if we had a perfect router model), as there were 1431 subjects in this test set, and in total 1152 of them were classified correctly by at least one classifer, which gives $1152/1431=80.5$% theoretical upper bound for a router between these models. It's hard to build a better model, perhaps it's easier to build a good router?

![Venn Diagram](https://github.com/trialan/eeg/assets/16582240/ea76f743-e977-4fb2-b42c-bad56752a367)

Actually the above venn diagram (generated using `venn3` is wrong, I'm not sure what it's counting but it has more subjects than there are in the validation set used to generate it, so something is wrong). Correct venn diagrams generated using `venn` are below for 3 and 4 model routing. Upper bounds are 82% and 88% respectively.

![3-model-Venn (corrected)](https://github.com/trialan/eeg/assets/16582240/706ea99b-a63a-4755-824a-bf71c2a2f9ed)
![4-model-Venn](https://github.com/trialan/eeg/assets/16582240/cc6db827-2072-458b-8e97-e0d6b1a0dfdb)

### Building a better router
Given the theoretical upper bounds for performance on this problem if we had a perfect router, it seems worth it to work on improving the router. Here is a first investigation:

![router_F(N)_full_plot](https://github.com/trialan/eeg/assets/16582240/2abec77f-cad7-4e7d-b53c-6cceefca6fc8)


What is very surprising is that when I run `router.py` and try multiple different routers, the best router doesn't result in the best final classification score. Why is that? Makes no sense. Still a lot to figure out about these routers.

```python
###### CSP+LDA Router score: 0.4166666666666667

###### Meta-clf score (CSP+LDA router): 0.6340782122905028

###### EDFgMDM Router score: 0.4041666666666667

###### Meta-clf score (EDFgMDM router): 0.6634078212290503
```


### CNNs with time series
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

And this gave the result $$0.6929 \pm 0.0001$. The magnitude on this is a bit hard to interpret but smaller numbers are better, and more data had a smaller number, so the Andrew Ng flowchart gave us the right idea and we will now experiment with data augmentation. Let's 3x the data using jittering. Now, in `jitter_augment.py` we augment the number of electrodes. Here we have evidence for augmenting the number of samples, so let's do that instead. I use the `augment_data` function in `cnn.py`. This is what I get:

![3x_jitter_CNN](https://github.com/trialan/eeg/assets/16582240/fd89b2ff-3a18-4739-addc-0a555080dbe7)

```python
In [1]: np.mean([np.min(v) for v in split_val_losses])
Out[1]: 0.6930747129882157

In [2]: np.std([np.min(v) for v in split_val_losses])/np.sqrt(5)
Out[2]: 3.35874317722525e-05
```

--> This didn't help. That's a bit odd, needs further reflecting.

### Fourier transforming coefficient matrix
![10subjects_3rdEigenmode_AverageOfFourierTransform](https://github.com/trialan/eeg/assets/123100675/d06ca0df-3b80-45f5-b45b-e6acbc8895c9)
The Fourier transforms of the coefficient of the third eigenmode as a function of time over each epoch in category '0' (probably 'hands') and category '1' for 10 subjects was taken. Then those fourier transforms were averaged. We see two consistent features, a dip in power around 0.1 for '1' and a difference in slope around the tails.
Here are the fourier transforms of eigenmode decomposition coefficients for the first 20 eigenmodes, orange is for hands, blue is for feet (or vice versa). The FTs for all 'hands' epochs were averaged (for all subjects) and same for feet. Notice eigenmode 16 - it might be used for distinguishing between the two?
![fourier_galore_allmodes](https://github.com/trialan/eeg/assets/123100675/d37d97f0-bc04-4206-b7e0-ddeb80e4031c)


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

