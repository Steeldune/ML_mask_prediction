import glob
import cv2
import numpy as np

image_folder = 'X:\\BEP_data\\Data_External\\RL015'

image_list = glob.glob(image_folder + '\\PPA\\*.png')
for im_str in image_list:
    im_name = im_str.split('\\')[-1].split('.')[0].split('_')

    im_name = [int(i) for i in im_name]

    im = cv2.imread(im_str)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #
    # cv2.imwrite(image_folder + '\\Collected\\{}_{}_{}_3.png'.format(im_str.split('\\')[-2], str(im_name[0]), str(im_name[1])), im)


    im_up = cv2.resize(im, (2048, 2048))

    im_1 = im_up[:1024, :1024]
    im_2 = im_up[1024:, :1024]
    im_3 = im_up[:1024, 1024:]
    im_4 = im_up[1024:, 1024:]
    cv2.imwrite(image_folder + '\\PPA_Upscale\\{}_{}_{}_2.png'.format(im_name[0], str(im_name[1]*2), str(im_name[2]*2)), im_1)
    cv2.imwrite(image_folder + '\\PPA_Upscale\\{}_{}_{}_2.png'.format(im_name[0], str(im_name[1]*2 + 1), str(im_name[2]*2)), im_2)
    cv2.imwrite(image_folder + '\\PPA_Upscale\\{}_{}_{}_2.png'.format(im_name[0], str(im_name[1]*2), str(im_name[2]*2+1)), im_3)
    cv2.imwrite(image_folder + '\\PPA_Upscale\\{}_{}_{}_2.png'.format(im_name[0], str(im_name[1]*2+1), str(im_name[2]*2+1)), im_4)
