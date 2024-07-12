# Turning measurements on the scalp into measurements on the brain

Classifying motor imagery with [ECoG](https://en.wikipedia.org/wiki/Electrocorticography) is [easy, achieving over 99.5% accuracy](https://sci-hub.se/10.1007/s42452-020-2023-x). So if we could transform our EEG data into ECoG data, we could crush the problem. This folder contains code related to this effort.

Ideally we'd have a `scalp_to_brain_transform` function that takes in physionet EEG data, transforms in into ECoG data, and then runs an LSTM classifier of the same kind that achieves such good accuracy as in the paper.
