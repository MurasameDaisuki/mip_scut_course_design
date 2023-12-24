import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using SimpleITK N4BiasFieldCorrection.
    :param in_file: .nii.gz 文件的输入路径
    :param out_file: .nii.gz 校正后的文件保存路径
    :return: 校正后的nii文件全路径名

    """
    # 使用SimpltITK N4BiasFieldCorrection校正MRI图像的偏置场
    input_image = sitk.ReadImage(in_file, image_type)
    output_image_s = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    sitk.WriteImage(output_image_s, out_file)
    image_ct = nib.load(out_file)
    image_ct_data = image_ct.get_fdata()
    print(in_file,image_ct_data.shape)
    return os.path.abspath(out_file)

path = r'dataset/image/'
save_path = r'dataset/image_process/'
path_list = os.listdir(r'E:\mip_cd\SpineSagT2Wdataset3\test\image')

for i in path_list:
    correct_bias(path+i, save_path+i, image_type=sitk.sitkFloat64)
