import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import get_envelope


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def check_overlap(x1, x2, y1, y2):
    overlap = np.sign(min(x2, y2) - max(x1, y1))
    if overlap == 1:
        return True
    else:
        return False


def exportNuclei(nuclei, ini_data_path, image, dilate=10, Zlevel=1, upscaleTo=None, thresh_mask=False):
    zoom_level = int(image.split('_')[2])
    if upscaleTo is None:
        upscaleTo = zoom_level
    export_path = ini_data_path + 'nuclei_exports/' + str(Zlevel) + '/' + image + '/'
    EM_path = ini_data_path + 'EM/' + str(Zlevel) + '/' + image + '.png'
    EM_image = cv2_imread(EM_path)
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    nuc_txt = open(export_path + 'border_coords.txt', 'w+')
    new_borders = np.zeros((np.shape(nuclei)[0], 4), dtype=int)

    for i, nucleus in enumerate(nuclei):
        new_borders[i] = (max(nucleus[1] - dilate, 0),
                          max(nucleus[0] - dilate, 0),
                          min(nucleus[1] + nucleus[3] + dilate, len(EM_image) - 1),
                          min(nucleus[0] + nucleus[2] + dilate, len(EM_image) - 1))

        if new_borders[i][0] * new_borders[i][1] == 0 or new_borders[i][2] == len(EM_image) - 1 or new_borders[i][
            3] == len(
                EM_image) - 1:
            nuc_txt.write('{}.{}\n'.format(i, new_borders[i]))

    if upscaleTo == zoom_level:
        for i, bound_box in enumerate(new_borders):
            # The bound box variable creates the corners of the box that will crop the EM image. It is made by comparing
            # the found crop coordinates plus a dilate value with the image borders. This way, when the bounding box
            # exceeds the image boundary, the program will take the image boundary as crop coordinate

            nucleus_crop = EM_image[bound_box[0]:bound_box[2], bound_box[1]: bound_box[3]]
            cv2.imwrite(export_path + 'n' + str(i) + '.png', nucleus_crop)

            # This checks if the bounding box overlaps with the image boundary, and writes the coords to a txt file for
            # later stitching.

            if bound_box[0] * bound_box[1] == 0 or bound_box[2] == len(EM_image) - 1 or bound_box[3] == len(
                    EM_image) - 1:
                nuc_txt.write('{}.{}\n'.format(i, bound_box))

    else:

        nuclei_exports = get_envelope.data_upscale(ini_data_path, zoom_level, upscaleTo, image, new_borders,
                                                  Zlevel=Zlevel, thresh_mask=thresh_mask)
        for i, nucleus in enumerate(nuclei_exports):
            cv2.imwrite(export_path + 'n' + str(i) + '.png', nucleus)
    nuc_txt.close()


def stitch_images(image1, coords1, image2, coords2, ini_path, Zlevel=1, Hor=True, UpscaleTo=None, thresh_mask=False):
    if UpscaleTo is None:
        EM_1 = cv2_imread(ini_path + 'EM/' + str(Zlevel) + '/' + image1 + '.png')
        EM_2 = cv2_imread(ini_path + 'EM/' + str(Zlevel) + '/' + image2 + '.png')
        EM_1_crop = EM_1[coords1[0]:coords1[2], coords1[1]: coords1[3]]
        EM_2_crop = EM_2[coords2[0]:coords2[2], coords2[1]: coords2[3]]
    else:
        zoom_level = int(image1.split('_')[2])
        EM_1_crop = get_envelope.data_upscale(ini_path, zoom_level, UpscaleTo, image1, [coords1], Zlevel=Zlevel,
                                             thresh_mask=thresh_mask)
        EM_2_crop = get_envelope.data_upscale(ini_path, zoom_level, UpscaleTo, image2, [coords2], Zlevel=Zlevel,
                                             thresh_mask=thresh_mask)
        zoom_factor = np.power(2, zoom_level - UpscaleTo)
        coords1 = [x * zoom_factor for x in coords1]
        coords2 = [x * zoom_factor for x in coords2]

    if Hor:
        stitched = np.ones((max(coords1[2], coords2[2]) - min(coords1[0], coords2[0]),
                            coords2[3] + coords1[3] - coords1[1] - coords2[1]), dtype=int)
        coords1_adj = [coords1[0] - min(coords1[0], coords2[0]), coords2[3] - coords2[1], 0, np.shape(stitched)[1]]
        coords2_adj = [coords2[0] - min(coords1[0], coords2[0]), 0, 0, coords2[3] - coords2[1]]
        coords1_adj[2] = coords1[2] - coords1[0] + coords1_adj[0]
        coords2_adj[2] = coords2[2] - coords2[0] + coords2_adj[0]

    else:
        stitched = np.ones((coords2[2] + coords1[2] - coords1[0] - coords2[0],
                            max(coords2[3], coords1[3]) - min(coords1[1], coords2[1])), dtype=int)
        coords1_adj = [coords2[2] - coords2[0], coords1[1] - min(coords1[1], coords2[1]), np.shape(stitched)[0], 0]
        coords2_adj = [0, coords2[1] - min(coords1[1], coords2[1]), coords2[2] - coords2[0], 0]
        coords1_adj[3] = coords1_adj[1] + (coords1[3] - coords1[1])
        coords2_adj[3] = coords2_adj[1] + (coords2[3] - coords2[1])
    if EM_2_crop and EM_1_crop is not None:
        stitched[coords2_adj[0]:coords2_adj[2], coords2_adj[1]:coords2_adj[3]] = EM_2_crop[0].astype(int)
        stitched[coords1_adj[0]:coords1_adj[2], coords1_adj[1]:coords1_adj[3]] = EM_1_crop[0].astype(int)
    export_path = ini_path + 'nuclei_exports/' + str(Zlevel) + '/stitched_images/'
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    cv2.imwrite(export_path + '{}and{}.png'.format(image2, image1), stitched)


def find_matches(ini_data_path, image_list, Zlevel=1, UpscaleTo=None, thresh_mask=False):
    for section in image_list:
        bordersRead = open(ini_data_path + 'nuclei_exports/' + str(Zlevel) + '/' + section + '/border_coords.txt', 'r')
        border_list = bordersRead.readlines()

        # Checks for border matches with the image to the left and top of itself

        for line in border_list:
            nucleus, coordsStr = line.split('.')
            coords = list(map(int, coordsStr[1:-2].split()))
            image_name_array = list(map(int, section.split('_')))
            if coords[0] == 0:
                reqImage = str(image_name_array[0] - 1) + '_' + str(image_name_array[1]) + '_' + str(
                    image_name_array[2])
                if reqImage in image_list:
                    checkRead = open(
                        ini_data_path + 'nuclei_exports/' + str(Zlevel) + '/' + reqImage + '/border_coords.txt', 'r')
                    found = False
                    for line2 in checkRead.readlines():
                        nucleus2, coordsStr2 = line2.split('.')
                        coords2 = list(map(int, coordsStr2[1:-2].split()))
                        if check_overlap(coords[1], coords[3], coords2[1], coords2[3]) and coords2[2] == 1023:

                            found = True
                            stitch_images(section, coords, reqImage, coords2, ini_data_path, Zlevel=Zlevel, Hor=False,
                                          UpscaleTo=UpscaleTo, thresh_mask=thresh_mask)
                            continue

                else:
                    continue
            if coords[1] == 0:
                reqImage = str(image_name_array[0]) + '_' + str(image_name_array[1] - 1) + '_' + str(
                    image_name_array[2])
                if reqImage in image_list:
                    checkRead = open(
                        ini_data_path + 'nuclei_exports/' + str(Zlevel) + '/' + reqImage + '/border_coords.txt', 'r')
                    found = False
                    for line2 in checkRead.readlines():
                        nucleus2, coordsStr2 = line2.split('.')
                        coords2 = list(map(int, coordsStr2[1:-2].split()))
                        if check_overlap(coords[0], coords[2], coords2[0], coords2[2]) and coords2[3] == 1023:
                            found = True
                            print('Match found between nucleus {} of image {} and {} {}'.format(nucleus, section,
                                                                                                nucleus2, reqImage))
                            stitch_images(section, coords, reqImage, coords2, ini_data_path, Zlevel=Zlevel,
                                          UpscaleTo=UpscaleTo, thresh_mask=thresh_mask)
                            continue
                    if found == False:
                        print(
                            'No match between nucleus {} of image {} and a nucleus of image {}'.format(nucleus, section,
                                                                                                       reqImage))
                else:
                    continue


def ShowResults(imgs_path, ini_data_path, image_list, Zlevel=1, upscaleTo=None, threshold_masks=False):
    erode_kernel = np.ones((3, 3), np.uint8)
    for image in image_list:
        pred_image = cv2_imread(imgs_path + image + '.png')

        # Find contours
        thresh_img = cv2.adaptiveThreshold(pred_image, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, -50)

        # cv2.imshow('Threshold_image_{}'.format(image), thresh_img*255)
        # cv2.imshow('Regular_image{}'.format(image), pred_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        _, cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Get bounding rectangles for the scale and the particles
        thr_size = 500
        particles = [cv2.boundingRect(cnt) for cnt in cnts if cv2.contourArea(cnt) > thr_size]
        splotches = [cv2.boundingRect(cnt) for cnt in cnts if cv2.contourArea(cnt) < thr_size]

        # Iterate all particles, add label to input image

        print(particles)
        exportNuclei(particles, ini_data_path, image, Zlevel=Zlevel, upscaleTo=upscaleTo, thresh_mask=threshold_masks)
    find_matches(ini_data_path, image_list, Zlevel=Zlevel, UpscaleTo=upscaleTo, thresh_mask=threshold_masks)


if __name__ == '__main__':
    ShowResults('C:/Users/night_3ns60sk/OneDrive/Documenten/TU_algemeen/GPU_BEP_PRACTICE/model_results/',
                'C:/Users/night_3ns60sk/OneDrive/Documenten/TU_algemeen/GPU_BEP_PRACTICE/data/',
                get_envelope.get_data('C:/Users/night_3ns60sk/OneDrive/Documenten/TU_algemeen/GPU_BEP_PRACTICE/data/',
                                     1, 3, (1, 1, 4, 4), JustStrs=True)[1], Zlevel=1, upscaleTo=0,
                threshold_masks=False)
