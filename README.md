# CycleGAN_MRtoCT
This repository comprises an MR-CT image synthesis project using a type of Generative Adversarial Network called [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf). The project is intended to implement CycleGAN on a very large dataset of unregistered MR and CT images.

The project was for a student research assistant position in the [Molecular Imaging / Magnetic Resonance Technology Lab](https://mimrtl.wiscweb.wisc.edu/) at the University of Wisconsin-Madison. Scripts in this project were run on an on-campus NVIDIA DGX system.

### Nifti_Generator.py
NiftiGenerator is a tool to ingest Nifti images using Nibabel, apply basic augmentation, and utilize them as inputs to a deep learning model. See file for more details on functionality. The outputs from NiftiGenerator are fed into cyclegan_keras.py for training the model.

### cyclegan_keras.py
Implementation and training script for the CycleGAN model. The model receives generated images from Nifti_Generator.py for training.

### cyclegan_keras_evaluation.py
Evaluation script for the CycleGAN model. Loads the trained model and runs on an evaluation dataset.

### voxel_normalization.py
Ensures the voxels of MR and CT have the same spatial dimensions. Because the images are unregistered, the images' voxels do not all represent the same physical dimensions. Given MR and CT directories, this script resamples images such that the voxels represent the same physical dimensions across all images before input to the model.
