#!/usr/bin/env python
# coding: utf-8
# %%


from keras.layers import Input, Concatenate, Activation, Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D, Dropout
from keras.layers import BatchNormalization as BN
import numpy as np
import os
import nrrd
from keras import backend as keras
import tensorflow as tf
import shutil
import SimpleITK as sitk

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
image_W, image_H, img_frame, img_channels, n_cls = 128, 128, 32, 1, 2


def getUnet3D(img_frame, img_channels, n_cls):
    # keras.set_floatx('float16')
    input_shape = (img_frame, None, None, img_channels)  # input_shape = (img_frame,img_rows,img_cols,img_channels)
    inputs = Input(input_shape)
    # print(inputs)

    conv1 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BN(axis=-1)(conv1)
    conv1 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BN(axis=-1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BN(axis=-1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BN(axis=-1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BN(axis=-1)(conv4)
    # print("conv4 shape:", conv4.shape)
    drop4 = Dropout(0.5)(conv4)
    # print("drop4 shape:", drop4.shape)

    up5 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop4))
    # print("up5 shape:", up5.shape)
    merge5 = Concatenate(axis=4)([conv3, up5])
    conv5 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BN(axis=-1)(conv5)
    # print("conv5 shape:", conv5.shape)

    up6 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv5))
    # print("up6 shape:", up6.shape)
    merge6 = Concatenate(axis=4)([conv2, up6])
    conv6 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BN(axis=-1)(conv6)
    # print("conv6 shape:", conv6.shape)

    up7 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv6))
    # print("up7 shape:", up7.shape)
    merge7 = Concatenate(axis=4)([conv1, up7])
    conv7 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BN(axis=-1)(conv7)
    preds = Conv3D(n_cls, 1, kernel_initializer='he_normal')(conv7)
    # print("preds shape:", preds.shape)

    return inputs, preds


def bianli(image, w, h, f, n_cls):
    imageShape = image.shape
    W = imageShape[2]
    H = imageShape[3]
    F = imageShape[1]

    x0list = np.arange(0, W - w, w // 3 * 2)  # Form a sliding window list in the width direction
    y0list = np.arange(0, H - h, h // 3 * 2)  # Form a sliding window list in the height direction
    z0list = np.arange(0, F - f, f // 2)  # Form a sliding window list in the slice direction
    if not x0list[-1] == W - w:
        x0list = np.append(x0list, W - w)  # If not divided with no remainder, the last sliding window is not equidistant to the end

    if not y0list[-1] == H - h:
        y0list = np.append(y0list, H - h)

    if not z0list[-1] == F - f:
        z0list = np.append(z0list, F - f)

    weight = np.zeros(imageShape)
    probility_mask = np.zeros(np.append(image.shape[0:-1], n_cls))  # softmax value
    for x0 in x0list:
        for y0 in y0list:
            for z0 in z0list:
                image_patch = image[:, z0:z0 + f, x0:x0 + w, y0:y0 + h, :]
                probility_mask_patch = sess.run(pred_softmax, feed_dict={inputs: image_patch})
                probility_mask[:, z0:z0 + f, x0:x0 + w, y0:y0 + h, :] += probility_mask_patch
                weight[:, z0:z0 + f, x0:x0 + w, y0:y0 + h, :] += 1
    output_mask = np.argmax(probility_mask / weight, 4)
    return output_mask


def to_uint8(img, min_=-256, max_=2048):
    """ 
    Perform clipping and convert data-type of img to uint8
    Args:
        img: numpy.ndarray
        min_: int, min clipping value
        max_: int, max clipping value
    return:
    """
    if img.max() == img.min():
        img = np.zeros_like(img)
    else:
        img = np.clip(img, min_, max_)
        img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img


def GetLargestConnectedCompont(binarysitk_image):
    """
    Get the largest connected component of the input mask.
    Args:
        binarysitk_image: SimpleITK.Image
    Return:
        outmask_sitk: SimpleITK.Image
    """
    cc = sitk.ConnectedComponent(binarysitk_image)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, binarysitk_image)
    maxlabel = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            maxlabel = l
            maxsize = size
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    outmask[labelmaskimage == maxlabel] = 1
    outmask[labelmaskimage != maxlabel] = 0
    outmask_sitk = sitk.GetImageFromArray(outmask)
    outmask_sitk.SetDirection(binarysitk_image.GetDirection())
    outmask_sitk.SetSpacing(binarysitk_image.GetSpacing())
    outmask_sitk.SetOrigin(binarysitk_image.GetOrigin())
    return outmask_sitk


print("Loading model ...")
keras.set_session(sess)
inputs, preds = getUnet3D(img_frame, img_channels, n_cls)  # define network
pred_softmax = tf.nn.softmax(preds)  
pred_mask = tf.expand_dims(tf.cast(tf.argmax(pred_softmax, 4), tf.float32),-1)  

saver = tf.compat.v1.train.Saver()
init_op = tf.compat.v1.global_variables_initializer()
sess.run(init_op)
saver.restore(sess, './trained_model_6/train.ckpt')

print("Model loaded.")


def predict(ctm_file):
    with sess.as_default():
        # saver.restore(sess, './trained_model_1/train.ckpt')
        image, _ = nrrd.read(ctm_file)
        image = to_uint8(image)
        image = np.transpose(image, (2, 0, 1))  # After the dimension order is changed, the dimensions are: axial,sagital,coronal
        image = image[np.newaxis, :, :, :, np.newaxis]  # Add a channel dimention
        image = (image.astype(np.float32)) / 255 - 0.5  # todo hyx 20220729
        output_mask = bianli(image, image_W, image_H, img_frame, n_cls)

        label = output_mask[0].astype(np.int16)
        sitk_label = sitk.GetImageFromArray(label)
        sitk_label = GetLargestConnectedCompont(sitk_label)
        pred = sitk.GetArrayFromImage(sitk_label)
        pred = pred.transpose(1, 2, 0).astype(np.uint8)
        return pred

def sitk_onehot_transform( label_sitk, n_class, SegMetaData_dict) :
    """
    Convert label_sitk（Segmentation-label.nrrd）to seg_sitk（Segmentation.seg.nrrd).
    Args:
        label_sitk: SimpleITK.Image.
        n_class： total number of foreground categories.
        SegMetaData_dict: dict, meta informations of the Segmantation.seg.nrrd file.
    Return:
        seg_sitk: SimpleITK.Image.
    """
    label_arr = sitk.GetArrayFromImage(label_sitk)
    shape = list(label_arr.shape)
    shape = shape+[n_class]
    seg_arr = np.zeros(shape=shape,dtype=np.uint8)
    for i in range(n_class):
        seg_arr[...,i] = (label_arr==i+1).astype(np.uint8)
    seg_sitk = sitk.GetImageFromArray(seg_arr)
    seg_sitk.CopyInformation(label_sitk)
    for k,v in SegMetaData_dict.items():
        seg_sitk.SetMetaData(key=k,value=v)
    return seg_sitk


def save_nrrd(path, mask_np, n_class, img_sitk, SegMetaData_dict):
    """
    Save predicted mask (mask_np) as nrrd files.
    args:
        path： str, results will be saved in this folder.
        mask_np：numpy array, predicted mask
        n_class： total number of foreground categories.
        img_sitk： SimpleITK.Image, CTM image.
        SegMetaData_dict： dict, meta informations of the Segmantation.seg.nrrd file.
    """
    label = mask_np
    # generate label_pred_sitk for the 'Segmentation-label.nrrd' file
    label_sitk = sitk.GetImageFromArray(label.transpose(2,1,0).astype(np.uint8))# (np)XYZ -> (np)ZYX -> (sitk)XYZ
    label_sitk.CopyInformation(img_sitk)
                                                               
    # generate onehot-encoding label (i.e label_pred_onehot_sitk) for the 'Segmentation.seg.nrrd' file
    seg_sitk = sitk_onehot_transform( label_sitk, n_class=n_class, SegMetaData_dict=SegMetaData_dict)   
        
    # save to nrrd file
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join( path,'Segmentation-label.nrrd' ) 
    sitk.WriteImage(label_sitk, filepath)
    filepath = os.path.join( path,'Segmentation.seg.nrrd' ) 
    print('Save as {}'.format(filepath))
    sitk.WriteImage(seg_sitk, filepath)
    # copy image nrrd file to save_dir
    filepath = os.path.join( path,'CTM.nrrd' ) 
    sitk.WriteImage(img_sitk, filepath)
    print('Save as {}'.format(filepath))
    return label_sitk, seg_sitk

def inference(ctm_file, save_path, n_class, SegMetaData_dict):
    """"
    Perform inference and save the results as Segmentation-label.nrrd and Segmentation.seg.nrrd.
    Args：
        ctm_file：str, CTM file in nrrd format of which mask you want the network to predict.
        save_path：str, the predicted results are saved in this folder.
        n_class: int, total number of foreground categories.
        SegMetaData_dict: dict, meta informations of the Segmentation.seg.nrrd file
    Return:
        label_sitk: SimpleITK Image object, the image written as Segmentation-label.nrrd
        seg_sitk: SimpleITK Image object, the image written as Segmentation.seg.nrrd
    """
    print('Start inferring...')
    mask = predict(ctm_file)
    print('Sucessfully completed inference.')
    print('Save the predicted mask as nrrd files...')
    img_sitk = sitk.ReadImage(ctm_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    label_sitk, seg_sitk = save_nrrd(save_path, mask.squeeze(), n_class=n_class, img_sitk=img_sitk, SegMetaData_dict=SegMetaData_dict)
    return label_sitk, seg_sitk 

if __name__ == "__main__":
    # default meta informations
    SegMetaData_dict = {
        'ITK_InputFilterName': 'NrrdImageIO',
        'ITK_original_direction': '[UNKNOWN_PRINT_CHARACTERISTICS]',
        'ITK_original_spacing': '[UNKNOWN_PRINT_CHARACTERISTICS]',
        'NRRD_kinds[0]': 'domain',
        'NRRD_kinds[1]': 'domain',
        'NRRD_measurement frame': '[UNKNOWN_PRINT_CHARACTERISTICS]',
        
        'Segment0_ID': 'Segment_1',
        'Segment0_Name': 'dura',
        'Segment0_NameAutoGenerated': '0',
        'Segment0_ColorAutoGenerated': '1',

        'Segmentation_MasterRepresentation': 'Binary labelmap',
        'Segmentation_ContainedRepresentationNames': 'Binary labelmap|',
        'Segmentation_ReferenceImageExtentOffset': '0 0 0',
    }
    n_class = 1# Total number of foreground categories.
    ctm_file = "./724648/CTM.nrrd"
    save_path = "./724648_inference"
    label_sitk, seg_sitk = inference(ctm_file=ctm_file, save_path=save_path, n_class=n_class, SegMetaData_dict=SegMetaData_dict)


# %%




