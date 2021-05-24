import numpy as np
import numpy.linalg
import skimage
import cv2
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn.cluster import OPTICS, cluster_optics_dbscan, AgglomerativeClustering, KMeans
from collections import Counter
import tqdm
import data_retrievals
from glob import glob
from compare_images import output_IOU


def get_path(img_str, Zlevel=1, type='EM'):
    return_path = 'data/{}/{}/{}.png'.format(type, str(Zlevel), img_str)
    return return_path


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def add_generic_line(img, height, vert=False, color=(255, 255, 0), thickness=3):
    IMG_WIDTH, IMG_HEIGHT = np.shape(img)
    img = cv2.line(img, ((0, height), (height, 0))[vert],
                   ((IMG_WIDTH, height), (height, IMG_HEIGHT))[vert], color, thickness)
    return img


def gib_entropy(img, kernel):
    img = entropy(img, kernel)
    img -= min(img.reshape(-1))
    img = skimage.img_as_float32(img / (max(img.reshape(-1)) - min(img.reshape(-1))))
    return img


def get_cluster_center(in_list):
    if len(in_list) == 1:
        return in_list[0]
    pos_sum = np.sum(in_list, axis=0)
    return pos_sum / (np.shape(in_list)[0])


def get_image_from_coords(img, coords):
    return_img = img[coords[1]:(coords[1] + coords[3]), coords[0]:(coords[0] + coords[2])]
    return return_img


def add_threshold(img, size, c):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, size, c)
    return img


def get_nuclei_pos(ho, kernel, dist_thresh=125):
    ho = cv2.threshold(ho, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #
    # cv2.imshow('{}'.format(img_str), ho)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ho_op = cv2.morphologyEx(ho, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(ho_op, kernel, iterations=3)
    dist_trans = cv2.distanceTransform(ho_op, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_trans, 0.7 * dist_trans.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    contours, hierarchy = cv2.findContours(unknown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mass_centres = np.zeros((len(contours), 2), dtype=int)

    for i in range(0, len(contours)):
        M = cv2.moments(contours[i], 0)
        mass_centres[i, 0] = (int(M['m10'] / M['m00']))
        mass_centres[i, 1] = (int(M['m01'] / M['m00']))

    if np.shape(mass_centres)[0] == 1:
        return [get_cluster_center(mass_centres)], [1, 2]

    clust = AgglomerativeClustering(None, distance_threshold=dist_thresh)

    clust.fit(mass_centres)
    labels = clust.labels_
    cluster_pos = []

    for i in range(max(labels) + 1):
        pos_list = []
        for j in range(len(labels)):
            if labels[j] == i:
                pos_list.append(mass_centres[j])
        pos_list = np.array(pos_list)
        cluster_pos.append(get_cluster_center(pos_list))

    return cluster_pos, labels


def get_voronoi_pattern(nuclei_pos, resolution=1024, em_img=None, OUTPUT_LIST=False):
    test_div = cv2.Subdiv2D()
    test_div.initDelaunay((0, 0, resolution, resolution))
    test_div.insert(nuclei_pos)
    voronoi_list = test_div.getVoronoiFacetList([])
    if OUTPUT_LIST:
        return voronoi_list

    out_img = np.zeros((resolution, resolution), dtype=np.uint8)

    for i, facet in enumerate(voronoi_list[0]):
        facet = np.array(facet, np.int32)
        facet = facet.reshape((-1, 1, 2))
        cv2.polylines(out_img, [facet], True, 255, 5, cv2.LINE_AA)

    text_img = out_img
    for i in nuclei_pos:
        i = i.astype(int)
        cv2.circle(text_img, (i[0], i[1]), 25, 127, -1)
    cv2.imwrite('imgs/voronoi.png', text_img)

    if em_img is None:
        return out_img
    else:
        return out_img, np.maximum(out_img, em_img)


def color_k_means(f_list, cluster_nr=3):
    k_means_list = f_list.copy()
    k_means_list = np.delete(k_means_list, [0, 1], 1)

    kmeans = KMeans(n_clusters=cluster_nr, random_state=0).fit(k_means_list)
    labels = kmeans.labels_
    f_list = np.array(f_list)
    out_img = np.zeros((max(f_list[:, 0]) + 1, max(f_list[:, 1]) + 1), dtype=np.uint8)
    for l, f in zip(labels, f_list):
        out_img[f[0], f[1]] = l + 1
    return out_img


def gen_voronoi_masks(img, nuclei_pos, resolution=1024, entropy_EM=None, feature_last_img=None):
    voronoi = get_voronoi_pattern(nuclei_pos, OUTPUT_LIST=True)
    masks = []
    new_poss = []
    f_lists = []
    rect_list = []
    clustered_full = np.zeros((resolution, resolution), dtype=np.uint8)
    for i, facet in enumerate(tqdm.tqdm(voronoi[0])):
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        facet = np.array(facet, np.int32)
        mask = cv2.drawContours(mask, [facet], -1, (255, 255, 255), -1, cv2.LINE_8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(contours) > 1:
            for cnt in contours:
                if np.shape(cnt)[0] == 4:
                    continue
                else:
                    contours[0] = cnt
        rect = cv2.boundingRect(contours[0])
        rect_list.append(rect)
        x, y, w, h = rect
        croped = img[y:y + h, x:x + w].copy()
        cropmsk = mask[y:y + h, x:x + w].copy()
        cropEM = entropy_EM[y:y + h, x:x + w].copy()
        cropLap = feature_last_img[y:y + h, x:x + w].copy()
        new_pos = (int(voronoi[1][i][0]), int(voronoi[1][i][1]))
        masks.append(cv2.bitwise_and(img, img, mask=mask))
        new_poss.append(new_pos)

        f_list = []

        for j in range(np.shape(croped)[0]):
            for k in range(np.shape(croped)[1]):
                if cropmsk[j, k] != 255:
                    continue
                else:
                    f_list.append([j, k, croped[j, k], cropEM[j, k], cropLap[j, k]])
        f_lists.append(f_list)

        EM_clustered = color_k_means(f_list)

        clustered_full[y:y + h, x:x + w] += EM_clustered

    return clustered_full * int(255 / 3)


def get_floodfill(img, coords, margin=3):
    label_lists = []
    for coord in coords:
        coord_new = [int(coord[1]), int(coord[0])]
        label_list = []
        for x in range(coord_new[0] - margin, coord_new[0] + margin):
            for y in range(coord_new[1] - margin, coord_new[1] + margin):
                label_list.append(img[x, y])

        if most_frequent(label_list) == img[coord_new[0], coord_new[1]]:
            retval, new_img, mask, rect = cv2.floodFill(img, None, (coord_new[1], coord_new[0]), 0)
        else:
            l = label_list.index(most_frequent(label_list))
            label_x = l % (margin * 2)
            label_y = l // (margin * 2)
            retval, new_img, mask, rect = cv2.floodFill(img, None, (
            (label_x - margin) + coord_new[1], (label_y - margin) + coord_new[0]), 0)
    return new_img


def assemble_labels(img_voronoi, img_cells, img_EM=None, close_op=False, dilate_op=False, disksize=3):
    if close_op:
        img_cells = cv2.morphologyEx(img_cells, cv2.MORPH_CLOSE, disk(disksize))
    if dilate_op:
        img_cells = cv2.morphologyEx(img_cells, cv2.MORPH_DILATE, disk(disksize))

    img_voronoi = cv2.morphologyEx(img_voronoi, cv2.MORPH_DILATE, disk(7))

    cell_voronoi_intersect = cv2.bitwise_and(img_voronoi, img_cells)
    img_voronoi -= cell_voronoi_intersect

    if img_EM is None:
        output_weights = cv2.bitwise_or(img_voronoi, img_cells)
        output_mask = img_cells
    else:
        output_weights = np.dstack((img_EM * img_voronoi, img_EM * img_cells, img_EM))
        output_mask = np.dstack((img_EM, img_EM, img_EM * img_cells))
    return output_weights, output_mask


def write_weights_masks(EM_address, HO_address, dist, thr_size=1200, truncate=True):
    clust_rad = int(1000 / np.power(2, dist))
    img_HO = cv2_imread(HO_address)
    img_EM = cv2_imread(EM_address)
    # img_HO = cv2.cvtColor(img_HO, cv2.COLOR_BGR2GRAY)
    # img_EM = cv2.cvtColor(img_EM, cv2.COLOR_BGR2GRAY)
    if truncate:
        img_HO = add_threshold(img_HO, 701, -50)
        # cv2.imshow('{}'.format(img_str), img_HO)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    IMG_WIDTH, IMG_HEIGHT = np.shape(img_HO)

    img_EM_gauss = cv2.GaussianBlur(img_EM, (3, 3), 3)
    img_EM_bil_75 = cv2.bilateralFilter(img_EM, 7, 75, 75)

    # cv2.imwrite('imgs/bilateral.png', img_EM_bil_75)

    img_EM_entropy = entropy(img_EM_gauss, disk(7))
    img_EM_entropy = img_EM_entropy - np.min(img_EM_entropy)
    img_EM_entropy = img_EM_entropy / np.max(img_EM_entropy)
    img_EM_entropy = 255 * img_EM_entropy
    img_EM_entropy = img_EM_entropy.astype(np.uint8)

    img_EM_laplacian = cv2.Laplacian(img_EM_bil_75, cv2.CV_8U)

    # cv2.imwrite('imgs/laplacian.png', img_EM_laplacian)
    # cv2.imwrite('imgs/entropy.png', img_EM_entropy)

    cluster_poss, _ = get_nuclei_pos(img_HO, disk(3), dist_thresh=clust_rad)
    voronoi_img, em_voronoi = get_voronoi_pattern(cluster_poss, em_img=img_EM_bil_75)
    img_EM_clustered = gen_voronoi_masks(img_EM_bil_75, cluster_poss, entropy_EM=img_EM_entropy,
                                         feature_last_img=img_EM_laplacian)

    # cv2.imwrite('imgs/labels.png', img_EM_clustered)
    img_EM_clustered_floodfill = get_floodfill(img_EM_clustered, cluster_poss)
    img_EM_clustered_floodfill = np.where(img_EM_clustered_floodfill == 0, 1.0, 0.0)
    img_EM_clustered_floodfill = (img_EM_clustered_floodfill * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(img_EM_clustered_floodfill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.imwrite('imgs/clustered.png', img_EM_clustered_floodfill)

    particles = [cnt for cnt in cnts if cv2.arcLength(cnt, True) < thr_size]

    # particles = [cnt for cnt in particles if cv2.contourArea(cnt) < 10000]
    im = np.zeros((1024, 1024), dtype=np.uint8)
    for contour in particles:
        cv2.drawContours(im, [contour], -1, 255, -1)

    return assemble_labels(voronoi_img, im, close_op=True, dilate_op=True)


if __name__ == '__main__':
    database = 'RL010'

    threshold = False

    ini_data_path = 'X:\\BEP_data\\{}\\'.format(database)
    img_list = glob(ini_data_path + 'Hoechst_predicted\\Collected\\*.png')

    dist = 3.9

    for HO_address in img_list:
        img_str = HO_address.split('\\')[-1]
        if glob(ini_data_path + 'EM\\Collected\\{}'.format(img_str)) != []:
            print('Working on image {}!'.format(img_str))
            EM_address = ini_data_path + 'EM\\Collected\\{}'.format(img_str)
            weight, mask = write_weights_masks(EM_address, HO_address, dist, truncate=threshold)
            cv2.imwrite(ini_data_path + '\\Masks_Generated\\{}'.format(img_str), mask)
            #
            # man_mask_img = cv2_imread('X:\\BEP_data\\{}\\Manual Masks\\'.format(database) + img_str)
            #
            # overl_img = mask/255 * man_mask_img/255
            # overl_img = np.multiply(overl_img, 255.0)
            #
            # mask_overlap = np.dstack((man_mask_img, overl_img.astype(np.uint8), mask))
            # cv2.imwrite('X:\\BEP_data\\Predict_set\\Mask_overlaps\\' + 'overlap_' + img_str, mask_overlap)
            #
            # mask = np.multiply(mask, 1/255.0)
            #
            # EM_img = cv2_imread('X:\\BEP_data\\{}\\EM\\Collected\\'.format(database) + img_str)
            # HO_img = cv2_imread(HO_address)
            # masked_img = np.dstack((EM_img * (
            #             1 - (cv2.normalize(HO_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))) * (1 - mask),
            #                         EM_img * (1 - mask),
            #                         EM_img * (1 - (
            #                             cv2.normalize(HO_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)))))
            # # cv2.imshow('{}_{}'.format(Zlevel, img), masked_img/255)
            # # cv2.waitKey()
            # # cv2.destroyAllWindows()
            # cv2.imwrite('X:\\BEP_data\\Predict_set\\EM_overlay\\' + 'em_overlay_' + img_str, masked_img)
