import numpy as np
import cv2
from PIL import Image
import os


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def read_img_string(img_str):
    img_array = img_str.split('_')
    img_array = list(map(int, img_array))
    return img_array


def weird_compare(interval, start, end):
    if interval - (end - start) % interval <= start % interval:
        return 1
    else:
        return 0


def get_data(ini_path, Z, zoom, bound_coords, JustStrs=False):
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
    exports = data_upscale('C:/Users/night_3ns60sk/OneDrive/Documenten/TU_algemeen/GPU_BEP_PRACTICE/data/', 3, 0,
                           '2_2_3',
                           [(80, 140, 190, 300), (600, 450, 800, 550)], Zlevel=3)
    for img in exports:
        cv2.imshow('Image_1', img / 255)
        cv2.waitKey()
        cv2.destroyAllWindows()
