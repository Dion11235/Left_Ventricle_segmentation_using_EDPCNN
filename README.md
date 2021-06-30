I have tried to explain and study the nuts and bolts of this fascinating methodology the researchers have used in the paper mentioned below. and I have tried to summarize the functions in a ipynb notebook. I hope this will help.

# Introduction
Link to the reference paper: [End-to-end Learning of Convolutional Neural Net and Dynamic Programming 
for Left Ventricle Segmentation](https://arxiv.org/abs/1812.00328)

Link to the mother code: https://github.com/minhnhat93/EDPCNN

# Requirements
- Numpy
- Pytorch >= 0.4
- TensorboardX
- Shapely
- Matplotlib
- Scipy
- Scikit-image
- Opencv for python
- nibabel
- h5py
- warmup-scheduler

# How to run
- you can create a folder called `preproc_data` and download the dataset hdf5 file from this [Google Drive](https://drive.google.com/open?id=1B7JC3WVSq1CcPJmYc3RGfhVFL12BWNKJ) link.
- simply run the cells of `demo_UNet_LV_Segment.ipynb` and `demo_EDPCNN.ipynb` to run the experiments. you can edit the hyperparameters in the args Series cell.

# Note
- This code only works on GPUs, preferrably NVIDIA ones with at least 10GB of VRAM. For GPUs with less VRAM, lowering the batch size may help.
- Due to the non-deterministic nature of large matrices reduction operations on GPU, the results over multiple runs will be slightly different but they usually have very similar loss curves and final performance.
- Sometime the training of the original U-Net may diverge and never go above 20% dice score on train set with only 10 images, simply restart the run script if this occurs.

# Result
- Look into the report `EDPCNN_project_report.pdf` to find the results of the experiments and the details of the theoretical backbone and details of the ACDC dataset.
- One can save the predicted masks for the test data using unet and EDPCNN, into the folders `UNet_predictions` and `EDPCNN_predictions` by running the last cells of the ipynb notebooks.
- The loss and dice score curves can be found in the `visualize` folder.