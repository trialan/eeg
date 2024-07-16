# eeg
Analysis of electroencephalogram data from [physionet](https://physionet.org/about/database/) with a view to reproducing the classification results in [this paper](https://hal.science/hal-03477057/document?fbclid=IwZXh0bgNhZW0CMTAAAR3UzR91MfBHO73CSZWK6QTDI6t0cpbEQHrmT9r8Vazzl9lGhewVMDXYVOY_aem_PxbOW954AyHy0jTub2Wlvw), and then improving them.

## Structure of the repo
At the top of the repo, we keep the code for reproducing the plot from Xu et al. as well as code useful for all experiments. In `/inverseproblem`, we put our code related to solving the [EEG inverse problem](https://www.fieldtriptoolbox.org/workshop/baci2017/inverseproblem/), this is our current angle for improving on Xu et al.'s results, and is currently under development. In `/experiments` we keep all our experiments, in particular that is where we have `routing_models/meta_clf.py`, our current best performing classifier.


## Reproduction of figure 3(a) from Xu et al.
This plot is produced by running `plot_reproduction.py`.

<p align="center">
  <img src="https://github.com/trialan/eeg/assets/16582240/189a2ee0-9108-4e5a-901e-6096781a20f2" alt="overnight_run" width="45%" height="300px">
  <img src="https://github.com/trialan/eeg/assets/16582240/89a80153-5df3-4abc-8db3-b94622b26080" alt="Their version" width="45%" height="300px">
</p>

Notice how in our plot, CSP + LDA is a curve, in there's it's a horizontal line, I believe this is because the curve jumps up to a value after ~5 components and then stays flat, so they chose to just plot the horizontal line to keep the plot more readable.
