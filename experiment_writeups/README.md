# Experiment write-up

## Brain-geometry informed experiments

### Laplacian Spatial Patterns dimensionality reduction
In this experiment I want to: 

1. Re-write the (n_channels, n_times) sub-matrices in the eigenbasis.
2. Do the CSP trick of taking average power + scaling to drop the time dimension.
3. Apply a standard classifier like LDA.

This did very poorly without the scaling (peak score about 54%, which is almost as bad as random guessing). With the scaling it also has a bad score but has a very interesting shape:

![LSP+LDA (64)](https://github.com/trialan/eeg/assets/16582240/2269fc8e-9d92-4c52-b2f3-2b79d66caee4)

So perhaps this would benefit from Taken's theorem!

## ML experiments

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

