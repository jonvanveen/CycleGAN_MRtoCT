import nibabel as nib
import cv2
from glob import glob
import types
import os
import numpy as np
from scipy.ndimage import affine_transform
import nilearn

class NiftiGenerator:

    imageFiles = []
    augOptions = types.SimpleNamespace()

    def initialize(self, input_folder, augOptions=None):
        self.imageFiles = glob(os.path.join(input_folder,'*.nii.gz'),recursive=True) + glob(os.path.join(input_folder,'*.nii'),recursive=True)
        
        if augOptions is None:
            self.set_no_augOptions()
        else:
            self.augOptions = augOptions
            
        print( '{} datasets were found'.format(len(self.imageFiles)) )
        #print( self.imageFiles )
        
    def set_no_augOptions(self):
        # augmode
        ## choices=['mirror','nearest','reflect','wrap']
        ## help='Determines how the augmented data is extended beyond its boundaries. See scipy.ndimage documentation for more information'
        self.augOptions.augmode = 'reflect'
        # augseed
        ## help='Random seed (as integer) to set for reproducible augmentation'
        self.augOptions.augseed = 813
        # addnoise
        ## help='Add Gaussian noise by this (floating point) factor'
        self.augOptions.addnoise = 0 # 1e-9 threw 'output_shape' not define error
        # hflips
        ## help='Perform random horizontal flips'
        self.augOptions.hflips = True
        # vflips
        ## help='Perform random horizontal flips'
        self.augOptions.vflips = True
        # rotations
        ## help='Perform random rotations up to this angle (in degrees)'
        self.augOptions.rotations = 0
        # scalings
        ## help='Perform random scalings between the range [(1-scale),(1+scale)]')
        self.augOptions.scalings = 0
        # shears
        ## help='Add random shears by up to this angle (in degrees)'
        self.augOptions.shears = 0
        # translations
        ## help='Perform random translations by up to this number of pixels'
        self.augOptions.translations = 0

    def generate(self, img_size=(256,256), slice_samples=1, batch_size=16 ):
    
        while True:
          
              batch_X = np.zeros( [batch_size,img_size[0],img_size[1],slice_samples] )
          
              for i in range(batch_size):
                  # get a random subject
                  j = np.random.randint( 0, len(self.imageFiles) )              
                  currImgFile = self.imageFiles[j]
              
                  # load nifti header
                  img = nib.load( currImgFile )
              
                  imgShape = img.header.get_data_shape()
                  
                  # determine sampling range
                  if slice_samples==1:
                      z = np.random.randint( 0, imgShape[2]-1 )
                  elif slice_samples==3:
                      z = np.random.randint( 1, imgShape[2]-2 )
                  elif slice_samples==5:
                      z = np.random.randint( 2, imgShape[2]-3 )
                  elif slice_samples==7:
                      z = np.random.randint( 3, imgShape[2]-4 )                  
                  elif slice_samples==9:
                      z = np.random.randint( 4, imgShape[2]-5 )
                  
                  # get slices                  
                  imgSlices = img.slicer[:,:,z-slice_samples//2:z+slice_samples//2+1].get_fdata()
                                    
                  # resize to fixed size for model (note img is resized with CUBIC)
                  imgSlices = cv2.resize( imgSlices, dsize=img_size, interpolation = cv2.INTER_CUBIC)

                  # ensure 3D matrix
                  if imgSlices.ndim == 2:
                      imgSlices = imgSlices[...,np.newaxis]                 

                  # augmentation would happen here
                  # TODO Augmentation needs to validated
                  imgSlices = self.augment( imgSlices )
              
                  # put into data array for batch of samples
                  batch_X[i,:,:,:] = imgSlices
                      
              yield( batch_X )

              
    def augment( self, X ):        
        # use affine transformations as augmentation
        M = np.eye(3)
        # horizontal flips
        if self.augOptions.hflips:
            M_ = np.eye(3)
            M_[1][1] = 1 if np.random.random()<0.5 else -1
            M = np.matmul(M,M_)
        # vertical flips
        if self.augOptions.vflips:
            M_ = np.eye(3)
            M_[0][0] = 1 if np.random.random()<0.5 else -1
            M = np.matmul(M,M_)
        # rotations
        if np.abs( self.augOptions.rotations ) > 1e-2:
            rot_angle = np.pi/180.0 * np.random.randint(-np.abs(self.augOptions.rotations),np.abs(self.augOptions.rotations))
            M_ = np.eye(3)
            M_[0][0] = np.cos(rot_angle)
            M_[0][1] = np.sin(rot_angle)
            M_[1][0] = -np.sin(rot_angle)
            M_[1][1] = np.cos(rot_angle)
            M = np.matmul(M,M_)
        # shears
        if np.abs( self.augOptions.shears ) > 1e-2:
            rot_angle_x = np.pi/180.0 * np.random.randint(-np.abs(self.augOptions.rotations),np.abs(self.augOptions.rotations))
            rot_angle_y = np.pi/180.0 * np.random.randint(-np.abs(self.augOptions.rotations),np.abs(self.augOptions.rotations))
            M_ = np.eye(3)
            M_[0][1] = np.tan(rot_angle_x)
            M_[1][0] = np.tan(rot_angle_y)
            M = np.matmul(M,M_)                    
        # scaling (also apply specified resizing [--imsize] here)
        if np.abs( self.augOptions.scalings ) > 1e-4:
            init_factor_x = 1
            init_factor_y = 1
            if np.abs( self.augOptions.scalings ) > 1e-4:
                random_factor_x = np.random.randint(-np.abs(self.augOptions.scalings)*10000,np.abs(self.augOptions.scalings)*10000)/10000
                random_factor_y = np.random.randint(-np.abs(self.augOptions.scalings)*10000,np.abs(self.augOptions.scalings)*10000)/10000
            else:
                random_factor_x = 0
                random_factor_y = 0
            scale_factor_x = init_factor_x + random_factor_x
            scale_factor_y = init_factor_y + random_factor_y
            M_ = np.eye(3)
            M_[0][0] = scale_factor_x
            M_[1][1] = scale_factor_y
            M = np.matmul(M,M_)
        # translations
        if np.abs( self.augOptions.translations ) > 0:
            translate_x = np.random.randint( -np.abs( self.augOptions.translations ), np.abs( self.augOptions.translations ) )
            translate_y = np.random.randint( -np.abs( self.augOptions.translations ), np.abs( self.augOptions.translations ) )
            M_ = np.eye(3)
            M_[0][2] = translate_x
            M_[1][2] = translate_y
            M = np.matmul(M,M_)

        # now apply the transform
        X_ = np.zeros_like(X)

        for k in range(X.shape[2]):
            X_[:,:,k] = affine_transform( X[:,:,k], M, output_shape=X[:,:,k].shape, mode=self.augOptions.augmode )
        X = X_

        # optionally add noise
        if np.abs( self.augOptions.addnoise ) > 1e-10:
            noise_mean = 0
            noise_sigma = self.augOptions.addnoise
            noise = np.random.normal( noise_mean, noise_sigma, output_shape )
            for k in range(X.shape[2]):
                X[:,:,k] = X[:,:,k] + noise
                
        return X