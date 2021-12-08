"""
This script ensures the voxels of MR and CT have the same spatial dimensions. 
Takes in images from text file of inputs and generates voxel normalized 
.nii files from input. The output normalized images are then fed into 
Nifti_Generator.py for use in cyclegan_keras.py.
"""


import os
import numpy as np
import nibabel as nib
import nibabel.processing

# Collect .nii files
t1_bravo_mr_voxelnorm = []
t1_bravo_ct_voxelnorm = []

shapes1, shapes2, shapes3 = [], [], []

mr_files = open('/nii/t1_bravo_mr.txt')
mr_files_str = mr_files.read()
inputFilesMR = mr_files_str.split(",")
# trim last empty item
inputFilesMR = inputFilesMR[:len(inputFilesMR) - 1]

ct_files = open('/nii/t1_bravo_ct.txt')
ct_files_str = ct_files.read()
inputFilesCT = ct_files_str.split(",")
# trim last empty item
inputFilesCT = inputFilesCT[:len(inputFilesCT) - 1]

# Find mean of voxel sizes to use for resampling
for entry in inputFilesMR:
    nii = nib.load(entry)
    sx, sy, sz = nii.header.get_zooms()
    shapes1.append(sx)
    shapes2.append(sy)
    shapes3.append(sz)

for entry in inputFilesCT:
    nii = nib.load(entry)
    sx, sy, sz = nii.header.get_zooms()
    shapes1.append(sx)
    shapes2.append(sy)
    shapes3.append(sz)

voxel_size = [np.mean(shapes1), np.mean(shapes2), np.mean(shapes3)] 
print(voxel_size)

# Apply resampling based on average voxel size
for entry in inputFilesMR:
    nii = nib.load(entry)
    voxelnorm = nib.processing.resample_to_output(nii, voxel_size)
    filename = entry[39:] # trim input filename for correct saving
    path = '/nii/mr_voxelnorm/Voxel Normalized ' + filename
    print("\n", "Before voxel norm: ")
    print(nii.header.get_data_shape())
    print("After voxel norm: ")
    print(voxelnorm.header.get_data_shape())
    t1_bravo_mr_voxelnorm.append(path)
    nib.save(voxelnorm, path)

print("\n","--------","\n")

for entry in inputFilesCT:
    nii = nib.load(entry)
    voxelnorm = nib.processing.resample_to_output(nii, voxel_size)
    filename = entry[39:]
    path = '/nii/ct_voxelnorm/Voxel Normalized ' + filename
    print("\n", "Before voxel norm: ")
    print(nii.header.get_data_shape())
    print("After voxel norm: ")
    print(voxelnorm.header.get_data_shape())
    t1_bravo_ct_voxelnorm.append(path)
    nib.save(voxelnorm, path)

# Generate voxel normalized txt files
with open(os.path.join('/nii', 't1_bravo_mr_voxelnorm.txt'), 'w+') as filehandle:
	for mr_item in t1_bravo_mr_voxelnorm:
		filehandle.write(mr_item + ',')
		
with open(os.path.join('/nii', 't1_bravo_ct_voxelnorm.txt'), 'w+') as filehandle:
	for ct_item in t1_bravo_ct_voxelnorm:
		filehandle.write(ct_item + ',')
