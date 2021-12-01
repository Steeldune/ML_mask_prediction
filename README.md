# ML Mask Prediction
---

#### Overview
This is the implementation of both a mask prediction network and a mask generation algorithm

#### Installation
```python
pip install git+git://github.com/steeldune/ML_mask_prediction.git
```

### Jupyter Notebooks
The Jupyter Notebooks are the heart of this project. They import various functions from the .py files in the mask_prediction folder. 

#### Qu Iteration
Within the notebooks folder are the main files for this project. The start of the process is done with the
qu_iteration.ipynb file, which applies the Qu iteration method of predicting nucleus locations purely through training data with dots denoting said locations.

#### Qi Img Prep
The qu_img_prep.ipynb folder takes the prediction from the Qu Iteration folder, and creates the generated masks for the final step.

#### Train Model
The train_model.ipynb file is the most standard neural network file. It takes an input folder and a training data folder, and trains a neural network to replicate the training data.

#### Use Model
The use_model.ipynb folder takes a model created in either the Qu Iteration file or the Train Model file, and it takes an input folder to make a prediction on that input data.

### Python Files

Some of the files contained in this folder are actively used by the Jupyter Notebooks, and some of them contain older data that might not work anymore. 
All of these file should have at least some amount of documentation as to what the functions in these files do.

- adjust_masks.py contains functions that can randomly transform mask training data. It might be useful for data augmentation.
- apply_weights.py is used in the Jupyter Notebooks, and is essential to the Qu Iteration method. It takes the point annotations, and can create weighted training data from them.
- compare_images.py takes folders of both manual masks and prediction images, and outputs the IOU score for each image in these folders. It will only output scores for overlapping lists of images.
- data_retrievals.py assembles the input data from EM and FM data.
- get_envelope.py is unused code for now. If used properly, it can export upscaled images of the segmentation blobs.
- particle_analysis.py is unused code, that can take the predictions of a network, and retrieve higher resolution images of the segmentations. It can also work across image boundaries.
- start_over.py is an essential file for the mask generation method.
- unet_semantics.py is an essential file for both the Qu Iteration method and neural network training in general.
- upsample_HO.py is unused code that was mostly a scratch file for manipulating data. 