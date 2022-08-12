#    Copyright of 2022 Chang Gung Medical Foundation 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import csv
import glob
import tqdm
import datetime
import numpy as np
import SimpleITK as sitk
from scipy.spatial.distance import cdist
import scipy

def getOverlapMetrics(refImage, predictImage):
    # Note that for the overlap measures filter, because we are dealing with a single label we
    # use the combined, all labels, evaluation measures without passing a specific label to the methods.
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    # Overlap measures
    overlap_measures_filter.Execute(sitk.Cast(refImage, sitk.sitkUInt8), sitk.Cast(predictImage, sitk.sitkUInt8))
    jaccard = overlap_measures_filter.GetJaccardCoefficient()
    dice = overlap_measures_filter.GetDiceCoefficient()
    volume_similarity = overlap_measures_filter.GetVolumeSimilarity()
    false_negative = overlap_measures_filter.GetFalseNegativeError()
    false_positive = overlap_measures_filter.GetFalsePositiveError()

    # 1-false_negative -> precision, 1-false_positive -> recall/sensitivity
    return dice, jaccard, volume_similarity, 1 - false_negative, 1 - false_positive


def getDistanceMetrics(refImage, predictImage):
    # Note that for the overlap measures filter, because we are dealing with a single label we
    # use the combined, all labels, evaluation measures without passing a specific label to the methods.
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    # Hausdorff distance
    hausdorff_distance_filter.Execute(sitk.Cast(refImage, sitk.sitkInt32), sitk.Cast(predictImage, sitk.sitkInt32))
    hausdorff_distance = hausdorff_distance_filter.GetHausdorffDistance()

    # Symmetric surface distance measures
    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside
    # relationship, is irrelevant)
    reference_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(refImage, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(refImage)

    segmented_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(predictImage, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(predictImage)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)  # use this!!

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))

    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances

    mean_surface_distance = np.float64(np.mean(all_surface_distances))
    median_surface_distance = np.float64(np.median(all_surface_distances))
    std_surface_distance = np.float64(np.std(all_surface_distances))
    max_surface_distance = np.float64(np.max(all_surface_distances))

    return hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance


def getVoxelLevelErrorStat(refImage, predImage):
    ref_data = sitk.GetArrayFromImage(refImage)
    predData = sitk.GetArrayFromImage(predImage)

    interSect_data = ref_data * predData

    if np.count_nonzero(ref_data) == 0:
        recall = 1.0
    else:
        recall = 1.0 * np.count_nonzero(interSect_data) / np.count_nonzero(ref_data)

    if np.count_nonzero(predData) == 0:
        precision = 1.0
    else:
        precision = 1.0 * np.count_nonzero(interSect_data) / np.count_nonzero(predData)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, precision, f1


class WholeImageEval(object):
    def __init__(self, pred_gt, output_postfix, chest_opx, oar_type):
        self.pred_gt = pred_gt
        self.is_cuda = True
        self.output_postfix = output_postfix
        self.outCSV = self._init_csv()
        self.oar, self.idx_oar = self._oar_names(chest_opx, oar_type)

    def _init_csv(self):
        # Evaluation metrics
        firstRow = ['patientID', 'organ', 'dice', 'jaccard', 'h95', 'hausdorff',
                    'vol_sim', 'ASD', 'medianSD', 'recall_voxel', 'precision_voxel']
        # Initialize output csv file name
        now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        outCSV = './Metrics_' + self.output_postfix + '_' + now_time + '.csv'
        with open(outCSV, 'w') as out:
            c_write = csv.writer(out)
            c_write.writerow(firstRow)
        return outCSV

    def _oar_names(self, chest_opx, oar_type):
        if "opx" in chest_opx:
            if oar_type == "all":
                oar = {
                    "Bkg": 0,
                    "Brachial.L": 1,
                    "Brachial.R": 2,
                    "BasalGanglia.L": 3,
                    "BasalGanglia.R": 4,
                    "Const.I": 9,
                    "Const.M": 10,
                    "Const.S": 11,
                    "Epiglottis": 16,
                    "Eso": 17,
                    "Hippocampus.L": 18,
                    "Hippocampus.R": 19,
                    "Larynx": 23,
                    "OralCavity": 29,
                    "Submandibular.L": 35,
                    "Submandibular.R": 36,
                    "TemporalLobe.L": 37,
                    "TemporalLobe.R": 38,
                    "Thyroid.L": 39,
                    "Thyroid.R": 40,
                    "TMJoint.L": 41,
                    "TMJoint.R": 42,

                    "Cochlea.L": 7,
                    "Cochlea.R": 8,
                    "InnerEar.L": 12,
                    "InnerEar.R": 13,
                    "HypoThalamus": 20,
                    "LacrimalGland.L": 21,
                    "LacrimalGland.R": 22,
                    "OpticChiasm": 26,
                    "OpticNerve.L": 27,
                    "OpticNerve.R": 28,
                    "PinealGland": 32,
                    "Pituitary": 33,

                    "BrainStem": 5,
                    "Cerebellum": 6,
                    "Eye.L": 14,
                    "Eye.R": 15,
                    "Mandible.L": 24,
                    "Mandible.R": 25,
                    "SpineCord": 34,
                    "Parotid.L": 30,
                    "Parotid.R": 31,

                }
            elif oar_type == "anchor":
                oar = {
                    "Bkg": 0,
                    "BrainStem": 1,
                    "Cerebellum": 2,
                    "Eye.L": 3,
                    "Eye.R": 4,
                    "Mandible.L": 5,
                    "Mandible.R": 6,
                    "SpineCord": 7,
                    "Parotid.L": 8,
                    "Parotid.R": 9,
                }
            elif oar_type == "mid":
                oar = {
                    "Bkg": 0,
                    "Brachial.L": 1,
                    "Brachial.R": 2,
                    "BasalGanglia.L": 3,
                    "BasalGanglia.R": 4,
                    "Const.I": 5,
                    "Const.M": 6,
                    "Const.S": 7,
                    "Epiglottis": 8,
                    "Eso": 9,
                    "Hippocampus.L": 10,
                    "Hippocampus.R": 11,
                    "Larynx": 12,
                    "OralCavity": 13,
                    "Submandibular.L": 14,
                    "Submandibular.R": 15,
                    "TemporalLobe.L": 16,
                    "TemporalLobe.R": 17,
                    "Thyroid.L": 18,
                    "Thyroid.R": 19,
                    "TMJoint.L": 20,
                    "TMJoint.R": 21,
                }
            else:
                oar = {
                    "Bkg": 0,
                    "Cochlea.L": 1,
                    "Cochlea.R": 2,
                    "InnerEar.L": 3,
                    "InnerEar.R": 4,
                    "HypoThalamus": 5,
                    "LacrimalGland.L": 6,
                    "LacrimalGland.R": 7,
                    "OpticChiasm": 8,
                    "OpticNerve.L": 9,
                    "OpticNerve.R": 10,
                    "PinealGland": 11,
                    "Pituitary": 12,
                }
        idx_oar = {}
        for k in oar:
            idx_oar[oar[k]] = k
        return oar, idx_oar

    def _get_stats(self, tmp_image_ref, tmp_image_pred, tmp_data_pred):
        dice, jaccard, vol_sim, precision_2, recall_2 = getOverlapMetrics(tmp_image_ref, tmp_image_pred)
        total_pred = np.sum(tmp_data_pred)
        if total_pred == 0:  # check if there is no prediction
            hd_sd, ave_sd, median_sd, std_sd, max_std = 999, 999, -1, -1, -1
            recall_voxel, precision_voxel, f1_voxel = -1, -1, -1
            h95 = -1
        else:
            hd_sd, ave_sd, median_sd, std_sd, max_std = getDistanceMetrics(tmp_image_ref, tmp_image_pred)
            recall_voxel, precision_voxel, f1_voxel = getVoxelLevelErrorStat(tmp_image_ref, tmp_image_pred)
            if hd_sd > 300:
                h95 = -1
            else:
                h95 = self._getHausdorff(tmp_image_pred, tmp_image_ref)
            h95 = -1
        return [dice, jaccard, h95, hd_sd, vol_sim, ave_sd, median_sd, recall_voxel, precision_voxel]

    def _getHausdorff(self, testImage, resultImage):
        """Compute the Hausdorff distance."""
        # Hausdorff distance is only defined when something is detected
        resultStatistics = sitk.StatisticsImageFilter()
        resultStatistics.Execute(resultImage)
        if resultStatistics.GetSum() == 0:
            return -1
        # resultStatistics.Execute(testImage)
        # if resultStatistics.GetSum() == 0:
        #     return -1
        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
        eTestImage = sitk.BinaryErode(testImage, (1, 1, 0))
        eResultImage = sitk.BinaryErode(resultImage, (1, 1, 0))
        hTestImage = sitk.Subtract(testImage, eTestImage)
        hResultImage = sitk.Subtract(resultImage, eResultImage)
        hTestArray = sitk.GetArrayFromImage(hTestImage)
        hResultArray = sitk.GetArrayFromImage(hResultImage)
        # Convert voxel location to world coordinates. Use the coordinate system of the test image
        # np.nonzero   = elements of the boundary in numpy order (zyx)
        # np.flipud    = elements in xyz order
        # np.transpose = create tuples (x,y,z)
        # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
        testCoordinates = np.apply_along_axis(testImage.TransformIndexToPhysicalPoint, 1,
                                              np.transpose(np.flipud(np.nonzero(hTestArray))).astype(int))
        resultCoordinates = np.apply_along_axis(testImage.TransformIndexToPhysicalPoint, 1,
                                                np.transpose(np.flipud(np.nonzero(hResultArray))).astype(int))

        # Use a kd-tree for fast spatial search
        def getDistancesFromAtoB(a, b):
            kdTree = scipy.spatial.KDTree(a, leafsize=100)
            return kdTree.query(b, k=1, eps=0, p=2)[0]

        # Compute distances from test to result; and result to test
        dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
        dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
        return max(round(np.percentile(dTestToResult, 95), 4), round(np.percentile(dResultToTest, 95), 4))

    def eval_img(self):
        for k in tqdm.tqdm(self.pred_gt.keys()):
            fname_pred = self.pred_gt[k][0]
            fname_ref = self.pred_gt[k][1]
            image_pred = sitk.ReadImage(fname_pred)
            image_ref = sitk.ReadImage(fname_ref)
            data_pred = sitk.GetArrayFromImage(image_pred)
            data_ref = sitk.GetArrayFromImage(image_ref)

            origin = image_ref.GetOrigin()
            direction = image_ref.GetDirection()
            spacing = image_ref.GetSpacing()

            labels = np.unique(data_ref)
            for j in labels:
                if j == 0:  # ignore the background class
                    continue
                else:
                    # get prediction binary mask SITK image for each OAR
                    tmp_data_ref = np.zeros(data_ref.shape, dtype=np.uint8)
                    ref_idx = data_ref == j
                    if np.sum(ref_idx) == 0:
                        continue
                    tmp_data_ref[ref_idx] = 1
                    tmp_image_ref = sitk.GetImageFromArray(tmp_data_ref)
                    tmp_image_ref.SetOrigin(origin)
                    tmp_image_ref.SetDirection(direction)
                    tmp_image_ref.SetSpacing(spacing)

                    # get gt binary mask SITK image for each OAR
                    tmp_data_pred = np.zeros(data_ref.shape, dtype=np.uint8)
                    tmp_data_pred[data_pred == j] = 1
                    tmp_image_pred = sitk.GetImageFromArray(tmp_data_pred)
                    tmp_image_pred.SetOrigin(origin)
                    tmp_image_pred.SetDirection(direction)
                    tmp_image_pred.SetSpacing(spacing)

                    oar_stats = self._get_stats(tmp_image_ref, tmp_image_pred, tmp_data_pred)
                    # output to csv file
                    patientID = fname_pred.split("/")[-1][:-7]
                    outputrow = [patientID, self.idx_oar[j], str(oar_stats[0]), str(oar_stats[1]),
                                 str(oar_stats[2]), str(oar_stats[3]), str(oar_stats[4]), str(oar_stats[5]),
                                 str(oar_stats[6]), str(oar_stats[7]), str(oar_stats[8])]
                    with open(self.outCSV, 'a') as out:
                        c_write = csv.writer(out)
                        c_write.writerow(outputrow)



def main():
    prefix_gt = ""
    postfix_gt = ".nii.gz"
    prefix_pred = ""
    postfix_pred = ".nii.gz"

    pth_gts = ["PATH_OF_ALL_GT",
               "PATH_OF_ANCHOR_GT",
               "PATH_OF_MID_GT",
               "PATH_OF_SMALLHARD_GT"]
    pth_preds = ["PATH_OF_ALL_PRED",
               "PATH_OF_ANCHOR_PRED",
               "PATH_OF_MID_PRED",
               "PATH_OF_SMALLHARD_PRED"]
    output_postfixs = ["ALL", "ANCHOR", "MID", "SMALLHARD"]
    oar_types = ["all", "anchor", "mid", "smallhard"]
    for i, pth_gt in enumerate(pth_gts):
        pth_pred = pth_preds[i]
        output_postfix = output_postfixs[i]
        oar_type = oar_types[i]
        files_gt = glob.glob(os.path.join(pth_gt, prefix_gt + "*" + postfix_gt))
        files_pred = glob.glob(os.path.join(pth_pred, prefix_pred + "*" + postfix_pred))

        kw_gt, kw_pred, pred_gt = {}, {}, {}
        for f in files_pred:
            kw = f.split("/")[-1][len(prefix_pred):-len(postfix_pred)]
            kw_pred[kw] = f
        for f in files_gt:
            kw = f.split("/")[-1][len(prefix_gt):-len(postfix_gt)]
            kw_gt[kw] = f
        for k in kw_pred.keys():
            if k in kw_gt.keys():
                pred_gt[k] = [kw_pred[k], kw_gt[k]]
        eval = WholeImageEval(pred_gt, output_postfix, "opx", oar_type)
        eval.eval_img()


if __name__ == "__main__":
    main()
