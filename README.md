# eeg
Analysis of electroencephalogram data from [physionet](https://physionet.org/about/database/) with a view to reproducing the classification results in [this paper](https://hal.science/hal-03477057/document?fbclid=IwZXh0bgNhZW0CMTAAAR3UzR91MfBHO73CSZWK6QTDI6t0cpbEQHrmT9r8Vazzl9lGhewVMDXYVOY_aem_PxbOW954AyHy0jTub2Wlvw), and then improving them.


### Reproduction of figure 3(a) from Xu et al.
This plot is produced by running `plot_reproduction.py`.

<p align="center">
  <img src="https://github.com/trialan/eeg/assets/16582240/189a2ee0-9108-4e5a-901e-6096781a20f2" alt="overnight_run" width="45%" height="300px">
  <img src="https://github.com/trialan/eeg/assets/16582240/89a80153-5df3-4abc-8db3-b94622b26080" alt="Their version" width="45%" height="300px">
</p>

Notice how in our plot, CSP + LDA is a curve, in their's it's a horizontal line, I believe this is because the curve jumps up to a value after ~5 components and then stays flat, so they chose to just plot the horizontal line to keep the plot more readable.


### Structure of the repo
At the top of the repo, we keep the code for reproducing the plot from Xu et al. as well as code useful for all experiments. In `/inverseproblem`, we put our code related to solving the [EEG inverse problem](https://www.fieldtriptoolbox.org/workshop/baci2017/inverseproblem/), this is our current angle for improving on Xu et al.'s results, and is currently under development. In `/experiments` we keep all our experiments, in particular that is where we have `routing_models/meta_clf.py`, our current best performing classifier.


### Inverse and forward problem

The interesting sources of EEG signal are located primarily on the cortical surface, we have plotted the triangulated pial surface from the MNE sample dataset (left). We then turn the mesh into a SpharaPy mesh so we can more easily work with it. We plot the dipole sources when making a bem model using the MNE package with 'oct4' spacing on our SpharaPy mesh (right). The spacing parameter determines how sparcely and in what geometrical patter dipoles are placed on the surface, taking a fraction of the vertices of the original mesh (right).
<p align="center">
  <img src="https://github.com/user-attachments/assets/82d2e22e-f72d-4daa-bc3e-8ef14992988a" width="45%" height="300px">
  <img src="https://github.com/user-attachments/assets/47c81e34-d60c-4017-8d17-a297d9231f5c" width="45%" height="300px">![sourcesonmesh](https://github.com/user-attachments/assets/47c81e34-d60c-4017-8d17-a297d9231f5c)

</p>


### References
Xiaoqi Xu, Nicolas Drougard, Raphaelle N Roy. Dimensionality Reduction via the Laplace-Beltrami
Operator: Application to EEG-based BCI. 2021 10th International IEEE/EMBS Conference on Neural
Engineering (NER) (2021)

Ou W, Hämäläinen MS, Golland P. A distributed spatio-temporal EEG/MEG inverse solver, Neuroimage (2009) 

R.G. Abeysuriya, P.A. Robinson, Real-time automated EEG tracking of brain states using neural field theory,
Journal of Neuroscience Methods (2015)


