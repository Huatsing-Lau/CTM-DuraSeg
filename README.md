# CTM-DuraSeg

Code for the paper: Automatic Segmentation of Dura for Quantitative of LUmar Stenosis: a deep learning study with 518 Myelograms. 

## Dependencies
This code depends on the follwing libraries:

* tensorflow
* keras
* SimpleITK
* pynrrd

## How to use:
Run the following command to obtain the prediction results in the form of nrrd file through the trained model(3D-Unet). You can open, view and modify this file on medical imaging software (such as 3D slicer).

~~~
python main_inference2nrrd.py
~~~

```python

```
