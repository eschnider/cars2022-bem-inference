#  Copyright (c) 2022. Eva Schnider

import argparse
import glob
import os
from typing import Union
from pathlib import Path
from timeit import default_timer as timer

import nibabel as nib
import numpy as np
from scipy import ndimage

from util.errors import NoConnectedComponentPresentError, TooManyComponentsPresentError


def save_nifty_with_new_ndarray(a: np.ndarray, nifty_destination_path: Union[str, os.PathLike],
                                nifty_file: nib.Nifti1Image):
    """
    Save a nifty file with data from an ndarray, but with the pre existing nifty affine.
    :param a: contains pixelwise data
    :param nifty_destination_path: full filename, where the new file will be saved
    :param nifty_file: a pre existing nifty file, we will use its affine.
    :return:
    """
    result = nib.Nifti1Image(a.astype(np.float64), affine=nifty_file.affine)
    nib.save(result, nifty_destination_path)


class Segment:
    UPPER_LIMIT = 100

    def __init__(self, segmentation_label: int, segmentation_array: np.ndarray):
        """
        :param segmentation_label: Integer label of the currently segmented strucuture
        :param segmentation_array: Binary segmentation of the currently selected structure
        """
        self.segmentation_label = segmentation_label
        vertebra_seg = np.zeros_like(segmentation_array)
        vertebra_seg[segmentation_array == segmentation_label] = 1
        self.blobs, self.num_features = ndimage.label(vertebra_seg)
        if self.num_features > self.UPPER_LIMIT:
            raise TooManyComponentsPresentError(
                f'Much more blobs than expected: {self.num_features}, have a look at label {segmentation_label}.')
        elif self.num_features == 0:
            raise NoConnectedComponentPresentError(f'No blobs for label {segmentation_label}.')
        self.unique_blob_indices, self.counts = np.unique(self.blobs, return_counts=True)
        self.biggest_blob_idx = self.unique_blob_indices[
            1 + np.argmax(self.counts[1:])]  # find biggest entry after the 0 (background)
        self.background_idx: int = 0
        main_blob = np.zeros_like(self.blobs)
        main_blob[self.blobs == self.biggest_blob_idx] = 1
        self.enlarged_main_blob = ndimage.binary_dilation(main_blob).astype(
            main_blob.dtype)


def relabel(data_path: Union[str, os.PathLike], destination_path: Union[str, os.PathLike]):
    """ Perform label correction with the label groups and upper limit as used in our publication. The code can be
    easily rewritten to compute the groups in parallel to save time. We left it in sequential order to improve
    readability.

    :param data_path: path to target .nii file
    :param destination_path: destination file path
    :return: None
    """
    start_time = timer()
    nifty_file = nib.load(data_path)
    nifty_data = nifty_file.get_fdata().astype(dtype=np.uint8)
    surrogate_label = 127  # should not be an actual label that is being used otherwise!

    corrected_segmentation = nifty_data.copy()
    last_time = timer()
    print(f'Postprocessing for {data_path}')
    print(f'Load nifty files: {timer() - last_time:.3}s.')

    last_time = timer()

    cervical_vertebrae_labels = [22, 23, 24, 25, 26, 27, 28]
    cervical_vertebrae_plus_overlap = cervical_vertebrae_labels + [1, 2]
    group_name = 'cervical vertebrae'
    for turn in range(1):
        corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation,
                                                                   labels_to_fix=cervical_vertebrae_plus_overlap)
        print(f'Fix {group_name} turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
        last_time = timer()

    thoracic_vertebrae_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    group_name = 'thoracic vertebrae'
    for turn in range(1):
        corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation,
                                                                   labels_to_fix=thoracic_vertebrae_labels)
        print(f'Fix {group_name} turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
        last_time = timer()

    lumbar_vertebrae_labels = [119, 120, 121, 122, 123]
    lumbar_vertebrae_plus_overlap_labels = [11, 12] + lumbar_vertebrae_labels
    group_name = 'lumbar vertebrae'
    for turn in range(1):
        corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation,
                                                                   labels_to_fix=lumbar_vertebrae_plus_overlap_labels)
        print(f'Fix {group_name} turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
        last_time = timer()

    big_bone_labels = [20, 21, 117, 115, 116, 118, 125]  # humerus, femur, coxae, sacrum
    group_name = 'big bones'
    for turn in range(1):
        corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation,
                                                                   labels_to_fix=big_bone_labels)
        print(f'Fix {group_name} turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
        last_time = timer()

    metacarpale_right_labels = [34, 36, 38, 40, 42]
    phalanx_proximalis_right_labels = [86, 92, 98, 104, 110]
    phalanx_media_right_labels = [90, 96, 102, 108]
    phalanx_distalis_right_labels = [84, 88, 94, 100, 106]
    group_name = 'hand right'
    hand_right_labels = metacarpale_right_labels + phalanx_proximalis_right_labels + phalanx_media_right_labels + phalanx_distalis_right_labels
    for turn in range(1):
        corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation,
                                                                   labels_to_fix=hand_right_labels)
        print(f'Fix {group_name} turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
        last_time = timer()

    metacarpale_left_labels = [33, 35, 37, 39, 41]
    phalanx_proximalis_left_labels = [85, 91, 97, 103, 109]
    phalanx_media_left_labels = [89, 95, 101, 107]
    phalanx_distalis_left_labels = [83, 87, 93, 99, 105]
    group_name = 'hand left'
    left_hand_labels = metacarpale_left_labels + phalanx_proximalis_left_labels + phalanx_media_left_labels + phalanx_distalis_left_labels
    for turn in range(1):
        corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation,
                                                                   labels_to_fix=left_hand_labels)
        print(f'Fix {group_name} turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
        last_time = timer()

    # ribs_right = [48, 56, 58, 60, 62, 64, 66, 68, 70, 50, 52, 54]
    group_name = 'ribs right'
    for turn in range(2):
        for idx, quartet in enumerate(
                [[48, 56, 58, 60], [58, 60, 62, 64], [62, 64, 66, 68], [66, 68, 70, 50],
                 [70, 50, 52, 54, surrogate_label]]):
            corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation, labels_to_fix=quartet)
            print(
                f'Fix {group_name}, group {idx}, turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
            last_time = timer()
    # ribs_left = [47, 55, 57, 59, 61, 63, 65, 67, 69, 49, 51, 53]
    group_name = 'ribs left'
    for turn in range(2):
        for idx, quartet in enumerate(
                [[47, 55, 57, 59], [57, 59, 61, 63], [61, 63, 65, 67], [65, 67, 69, 49],
                 [69, 49, 51, 53, surrogate_label]]):
            corrected_segmentation, change_counter = fix_segmentations(corrected_segmentation, labels_to_fix=quartet)
            print(
                f'Fix {group_name}, group {idx}, turn {turn}: {timer() - last_time:.3}s, changed {change_counter} blobs.')
            last_time = timer()
    save_nifty_with_new_ndarray(corrected_segmentation, destination_path, nifty_file)

    print(f'Total time: {timer() - start_time:.3}s')


def fix_segmentations(segmentation_array, labels_to_fix):
    segments = {}
    change_counter = 0
    labels_with_nonempty_blobs = []

    # initiate blobs for all labels that need fixing
    for vertebra_label in labels_to_fix:
        try:
            segments[vertebra_label] = Segment(vertebra_label, segmentation_array)
            labels_with_nonempty_blobs.append(vertebra_label)
        except (NoConnectedComponentPresentError, TooManyComponentsPresentError):
            pass

    for vertebra_label in labels_with_nonempty_blobs:
        current_segment = segments[vertebra_label]

        if current_segment.num_features > 1:  # more than background and one blob
            for blob_idx in set(current_segment.unique_blob_indices) - {current_segment.background_idx,
                                                                        current_segment.biggest_blob_idx}:  # we don't need the background and the biggest blob here
                stray_blob = np.zeros_like(segmentation_array)
                stray_blob[current_segment.blobs == blob_idx] = 1
                parent_vertebra_label = find_parent_blob(stray_blob, labels_with_nonempty_blobs, segments)
                if parent_vertebra_label > 0:
                    segmentation_array[stray_blob == 1] = parent_vertebra_label
                    change_counter = change_counter + 1

    return segmentation_array, change_counter


def find_parent_blob(child_blob, vertebra_labels, segments):
    for vertebra_label in vertebra_labels:
        # is there an intersection between the stray blob and the enlarged main blob of another vertebra?
        is_intersecting = np.any(np.logical_and(segments[vertebra_label].enlarged_main_blob == 1, child_blob == 1))
        if is_intersecting:
            return vertebra_label
    # nothing found
    return -1


if __name__ == '__main__':
    data_path = 'label_correction_demo_data/data_not_published.nii.gz'
    destination_path = 'label_correction_demo_data/data_not_published-corrected.nii.gz'

    relabel(data_path=data_path, destination_path=destination_path)
