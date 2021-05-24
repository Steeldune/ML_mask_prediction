import numpy as np
import cv2
import os
from data_retrievals import cv2_imread


def read_img_string(img_str):
    """
    :param      img_str: a string describing an image, like '4_3_3'
    :return:    integer array of this list. e.g: [4,3,3]
    """
    img_array = img_str.split('_')
    img_array = list(map(int, img_array))
    return img_array


def weird_compare(interval, start, end):
    """
    This is a function that is used later on in data_upscale, to add a corrective integer (either 1 or 0),
    when that function checks whether the image it is working on has reached the border.
    This function is not expected to be used anywhere outside data_upscale.
    :param interval:    Width in pixels of the target zoom images compared to the original image.
    :param start:       Position in pixels of the start of the ROI (either in X or Y direction)
    :param end:         Position in pixels of the end of the ROI (either in X or Y direction)
    :return:            0 or 1, depending on where the ROI lies on the direction.
    """
    if interval - (end - start) % interval <= start % interval:
        return 1
    else:
        return 0


def get_data(ini_path, Z, zoom, bound_coords, JustStrs=False):
    """
    This function either returns a list of names for the images that it gets the boundary coordinates from,
    or it gives the list, plus a 4-dimensional array of the data from those images.
    :param ini_path:    Address pointing to a dataset file containing EM, HO and IN images.
    :param Z:           Integer describing the layer from which to pull the images
    :param zoom:        Integer describing the level of detail to get (0 highest, 6 lowest)
    :param bound_coords:Start and end points of the images that are going to be pulled.
                        These coordinates point to the names of the images themselves.
    :param JustStrs:    Boolean on whether the function returns just the list of strings (True)
                        or if it returns the data as well (False)
    :return:            Either a list of strings, or the list plus an array of the data described in those strings.
    """
    EM_path = ini_path + 'EM/' + str(Z) + '/'
    HO_path = ini_path + 'Hoechst/' + str(Z) + '/'
    IN_path = ini_path + 'Insulin/' + str(Z) + '/'

    img_strs = []
    for x in range(bound_coords[2]):
        for y in range(bound_coords[3]):
            img_string = str(x + bound_coords[0]) + '_' + str(y + bound_coords[1]) + '_' + str(zoom)
            img_strs.append(img_string)

    if not JustStrs:
        data = np.zeros((bound_coords[2] * bound_coords[3], 1024, 1024, 3))

        for i, img in enumerate(img_strs):
            data_slice = np.zeros((1024, 1024, 3))
            data_slice[:, :, 0] = cv2_imread(EM_path + img + '.png')
            data_slice[:, :, 1] = cv2_imread(HO_path + img + '.png')
            data_slice[:, :, 2] = cv2_imread(IN_path + img + '.png')
            data[i] = data_slice
        return data, img_strs
    if JustStrs:
        return img_strs


def get_mask(ini_path, coords, img_str, zoom_factor, Zlevel=1, r=1, thresh=False):
    """
    Takes a given mask, and applies the appropriate crops and scaling to multiply it with the data it is supposed to mask.
    :param ini_path:    Address points to a folder containing Nuclei_masks
    :param coords:      Boundary points [x, y, x+dx, y+dy] which crops the mask.
    :param img_str:     string pointing to the specific mask that we want.
    :param zoom_factor: The zoom factor which will be applied to scale the mask.
    :param Zlevel:      The Z coordinate which will also be used to get at the mask you want.
    :param r:           The mask index for the image we're dealing with.
    :param thresh:      Boolean on whether to threshold the mask image.
    :return:            The cropped and scaled mask image.
    """
    mask_img = cv2_imread(ini_path + 'Nuclei_masks/' + str(Zlevel) + '/' + img_str + '.png')
    mask_crop = mask_img[coords[0]:coords[2], coords[1]: coords[3]]
    mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, np.ones((4, 4)))
    mask_crop_scale = cv2.resize(mask_crop,
                                 (np.shape(mask_crop)[1] * zoom_factor, np.shape(mask_crop)[0] * zoom_factor),
                                 interpolation=cv2.INTER_CUBIC)
    if thresh:
        mask_crop_scale = cv2.threshold(mask_crop_scale, 255 / 4, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(ini_path + 'nuclei_exports/' + str(Zlevel) + '/' + img_str + '/' + img_str + '_mask' + str(r) + '.png',
                mask_crop_scale)
    return mask_crop_scale


def data_upscale(ini_path, currentZoom, targetZoom, img_str, borderCoords, type='EM', Zlevel=1, use_mask=True,
                 thresh_mask=False):
    """
    This function takes one image, boundary coordinates on the pixel level and a target for the amount of detail,
    and delivers a cropped version of the original image, with the upscaled data.
    More explanation will follow in the function itself.
    :param ini_path:        Address pointing to a file which contains the various data types (EM, HO etc.).
    :param currentZoom:     The zoom level of the image from which you want more detail. (6 worst, 0 best)
    :param targetZoom:      The zoom level at which you want the detail to be (6 worst, 0 best)
    IMPORTANT: CurrentZoom HAS to be equal or larger than TargetZoom.
    :param img_str:         The particular image that you want to crop and upscale.
    :param borderCoords:    The coordinates in pixels which crop the image [y, x, y + dy, x + dx]
    :param type:            String which specifies the type of data to upscale (EM, HO, etc.)
    :param Zlevel:          The Z-level at which your data lies
    :param use_mask:        Boolean on whether to multiply the upscaled data with a corresponding mask.
    :param thresh_mask:     Boolean on whether to threshold the mask mentions one line above.
    :return:                The upscaled data in array form.
    """
    zoom_factor = np.power(2, currentZoom - targetZoom)

    img = read_img_string(img_str)

    interval = int(1024 / zoom_factor)
    nuclei_exports = []
    for r, coords in enumerate(borderCoords):
        no_image = False
        if use_mask:
            img_mask = get_mask(ini_path, coords, img_str, zoom_factor, Zlevel=Zlevel, r=r, thresh=thresh_mask) / 255
        y_range = range(img[0] * zoom_factor, img[0] * zoom_factor + zoom_factor)
        x_range = range(img[1] * zoom_factor, img[1] * zoom_factor + zoom_factor)
        y_range = y_range[coords[0] // interval: 1 + coords[2] // interval]
        x_range = x_range[coords[1] // interval: 1 + coords[3] // interval]
        img_data_list = []
        for i, x in enumerate(x_range):
            """The next two for loops go along all high-detail images that are involved in the crop. 
            The entire image is loaded, then it is cut down according to the boundary coordinates"""
            column_list = []
            for j, y in enumerate(y_range):
                target_img_str = str(y) + '_' + str(x) + '_' + str(targetZoom)
                img_data = []
                if not os.path.exists(ini_path + type + '/' + str(Zlevel) + '/' + target_img_str + '.png'):
                    print('No path found for image {}'.format(target_img_str))
                    no_image = True
                    break
                img_data = cv2_imread(ini_path + type + '/' + str(Zlevel) + '/' + target_img_str + '.png')

                if j == (weird_compare(interval, coords[0], coords[2]) + (coords[2] - coords[0]) // interval):
                    """This if statement checks if the image we're dealing with crosses with a boundary. 
                    If this is the case, the image will be cut off at that boundary. Same for the if statement below."""
                    img_data = img_data[: ((coords[2] % interval) * zoom_factor), :]
                if j == 0:
                    img_data = img_data[((coords[0] % interval) * zoom_factor):, :]
                    # cv2.imshow('test_1', img_data)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                column_list.append(img_data)

            if i == (weird_compare(interval, coords[1], coords[3]) + (coords[3] - coords[1]) // interval):
                for n, img_data in enumerate(column_list):
                    column_list[n] = img_data[:, : ((coords[3] % interval) * zoom_factor)]

            if i == 0:
                for n, img_data in enumerate(column_list):
                    column_list[n] = img_data[:, ((coords[1] % interval) * zoom_factor):]
            if not no_image:
                column_h = [np.shape(x)[0] for x in column_list]
                column = np.zeros((sum(column_h), np.shape(column_list[0])[1]), dtype=int)
                for k, [h, h_next] in enumerate(
                        zip(np.concatenate((0, np.cumsum(column_h)[:-1]), axis=None), np.cumsum(column_h))):
                    column[h:h_next, :] = column_list[k]
                img_data_list.append(column)
        if not no_image:
            row_w = [np.shape(x)[1] for x in img_data_list]
            row = np.zeros((np.shape(img_data_list[0])[0], sum(row_w)))
            for k, [w, w_next] in enumerate(
                    zip(np.concatenate((0, np.cumsum(row_w)[:-1]), axis=None), np.cumsum(row_w))):
                row[:, w: w_next] = img_data_list[k]
            # cv2.imwrite('stitched/upscaled{}_{}.png'.format(img, r), row)
            if use_mask:
                row = row * img_mask
            nuclei_exports.append(row)
    return nuclei_exports


if __name__ == '__main__':
    exports = data_upscale('X:\\BEP_data\\RL015\\', 6, 3 ,
                           '0_1_6',
                           [(858, 242, 858 + 128, 242 + 128)], Zlevel=1, use_mask=False, type='Hoechst')

    """"IMPORTANT: For manually writing down border coordinates, make sure to put it Y THEN X. 
     Reverse the order you would usually find them in image manipulation programs."""

    for img in exports:
        cv2.imshow('Image_1', img / 255)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imwrite('X:\\BEP_data\\RL015\\Crop_exports\\HO1.png', img)
