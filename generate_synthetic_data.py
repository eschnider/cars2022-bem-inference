#  Copyright (c) 2022. Eva Schnider

import os.path
from typing import Union

import nibabel as nib
import numpy as np
import scipy.ndimage
import skimage.morphology

from synthetic_data.labels import BodyPartLabelGroups, Labels
from synthetic_data.shape_creator import ShapeCreator


def create_segmentation_skeleton(size_lr: int = 64, size_is: int = 128, size_ap: int = 64, random_seed: int = 42
                                 ) -> np.ndarray:
    """ Build an artificial 3D skeleton with individual int labels per distinct bone.
    Outputs a 3D scalar volume of shape (size_lr, size_ap, size_is) where each voxel is a number indicating the type of
    tissue at that location.
    :param size_lr: number of pixels in left-right direction
    :param size_is: number of pixels in inferior-superior direction
    :param size_ap: number of pixels in anterior-posterior direction
    :param random_seed:
    :return: scalar volume of shape (size_lr, size_ap, size_is)
    """
    rng = np.random.default_rng(random_seed)
    scale_factor = np.ceil(size_lr / 8)

    data = np.zeros((size_lr, size_ap, size_is,), dtype=np.int16)
    shape_creator = ShapeCreator(size_lr, size_is, size_ap)

    random_lr_offset = rng.integers(-scale_factor, scale_factor, size=1)[0]
    random_ap_offset = rng.integers(-scale_factor, scale_factor, size=1)[0]
    random_is_offset = rng.integers(-scale_factor, scale_factor, size=1)[0]

    body_lr_center = size_lr / 2 + random_lr_offset
    body_ap_center = size_ap / 8 * 3 + random_ap_offset

    # Head
    skull_radius = rng.integers(7 / 8 * scale_factor, 10 / 8 * scale_factor, size=1)[0]
    skull_center_is = size_is - skull_radius * 2 - random_is_offset
    skull_mask = shape_creator.half_sphere_upper(center_lr=body_lr_center,
                                                 center_ap=body_ap_center,
                                                 center_is=skull_center_is,
                                                 radius=skull_radius)
    mandible_mask = shape_creator.half_sphere_lower(center_lr=body_lr_center,
                                                    center_ap=body_ap_center,
                                                    center_is=skull_center_is,
                                                    radius=skull_radius)
    # Spine
    vertebra_len_lr = vertebra_len_ap = 5 / 8 * scale_factor
    vertebra_c_len_is = rng.integers(3 / 8 * scale_factor, 4 / 8 * scale_factor, size=1)[0]
    vertebra_t_len_is = rng.integers(5 / 8 * scale_factor, 6 / 8 * scale_factor, size=1)[0]
    vertebra_l_len_is = rng.integers(5 / 8 * scale_factor, 6 / 8 * scale_factor, size=1)[0]

    c1_is_center = size_is - skull_radius * 4
    c2_is_center = c1_is_center - vertebra_c_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    t1_is_center = c2_is_center - vertebra_t_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    t2_is_center = t1_is_center - vertebra_t_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    t3_is_center = t2_is_center - vertebra_t_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    t4_is_center = t3_is_center - vertebra_t_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    l1_is_center = t4_is_center - vertebra_l_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    l2_is_center = l1_is_center - vertebra_l_len_is - rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[
        0]
    sacrum_is_center = l2_is_center - rng.integers(3 / 8 * scale_factor, 7 / 8 * scale_factor, size=1)[0]
    c1_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=c1_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_c_len_is)
    c2_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=c2_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_c_len_is)
    t1_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=t1_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_t_len_is)
    t2_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=t2_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_t_len_is)
    t3_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=t3_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_t_len_is)
    t4_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=t4_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_t_len_is)
    l1_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=l1_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_l_len_is)
    l2_mask = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center, center_is=l2_is_center,
                                  len_lr=vertebra_len_lr, len_ap=vertebra_len_ap, len_is=vertebra_l_len_is)
    sacrum_base_len = rng.integers(4 / 8 * scale_factor, 1 * scale_factor, size=1)[0]
    sacrum_heigth = rng.integers(5 / 8 * scale_factor, 1 * scale_factor, size=1)[0]
    sacrum_mask = shape_creator.pyramid_on_tip(center_base_lr=body_lr_center, center_base_ap=body_ap_center,
                                               center_base_is=sacrum_is_center, base_len=sacrum_base_len,
                                               height=sacrum_heigth)
    # Shoulders
    shoulder_width = rng.integers(15 / 8 * scale_factor, 20 / 8 * scale_factor, size=1)[0]
    clavicula_radius = 3 / 8 * scale_factor
    clavicula_lr_offset = 2 / 8 * scale_factor
    clavicula_left_mask = shape_creator.tube_horizontal(
        center_lr=body_lr_center - shoulder_width / 2 - vertebra_len_lr / 2 - clavicula_lr_offset,
        center_ap=body_ap_center,
        center_is=c1_is_center, len_lr=shoulder_width, radius=clavicula_radius)
    clavicula_right_mask = shape_creator.tube_horizontal(
        center_lr=body_lr_center + shoulder_width / 2 + vertebra_len_lr / 2 + clavicula_lr_offset,
        center_ap=body_ap_center,
        center_is=c1_is_center, len_lr=shoulder_width, radius=clavicula_radius)

    # Ribs

    rip_len_lr = rng.integers(7 / 8 * scale_factor, 9 / 8 * scale_factor, size=1)[0]
    rip_len_ap = rng.integers(7 / 8 * scale_factor, 9 / 8 * scale_factor, size=1)[0]
    rip_height = rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[0]
    # U shaped ribs
    rip_1_left = shape_creator.u_horizontal_open_right(center_lr=body_lr_center - rip_len_lr - vertebra_len_lr / 2,
                                                       center_ap=body_ap_center + rip_len_ap / 2,
                                                       center_is=t1_is_center, height=rip_len_lr, breadth=rip_len_ap,
                                                       thickness=rip_height)
    rip_1_right = shape_creator.u_horizontal_open_left(center_lr=body_lr_center + rip_len_lr + vertebra_len_lr / 2,
                                                       center_ap=body_ap_center + rip_len_ap / 2,
                                                       center_is=t1_is_center, height=rip_len_lr, breadth=rip_len_ap,
                                                       thickness=rip_height)
    rip_2_left = shape_creator.u_horizontal_open_right(center_lr=body_lr_center - rip_len_lr - vertebra_len_lr / 2,
                                                       center_ap=body_ap_center + rip_len_ap / 2,
                                                       center_is=t2_is_center, height=rip_len_lr, breadth=rip_len_ap,
                                                       thickness=rip_height)
    rip_2_right = shape_creator.u_horizontal_open_left(center_lr=body_lr_center + rip_len_lr + vertebra_len_lr / 2,
                                                       center_ap=body_ap_center + rip_len_ap / 2,
                                                       center_is=t2_is_center, height=rip_len_lr, breadth=rip_len_ap,
                                                       thickness=rip_height)
    rip_3_left = shape_creator.u_horizontal_open_right(center_lr=body_lr_center - rip_len_lr - vertebra_len_lr / 2,
                                                       center_ap=body_ap_center + rip_len_ap / 2,
                                                       center_is=t3_is_center, height=rip_len_lr, breadth=rip_len_ap,
                                                       thickness=rip_height)
    rip_3_right = shape_creator.u_horizontal_open_left(center_lr=body_lr_center + rip_len_lr + vertebra_len_lr / 2,
                                                       center_ap=body_ap_center + rip_len_ap / 2,
                                                       center_is=t3_is_center, height=rip_len_lr, breadth=rip_len_ap,
                                                       thickness=rip_height)
    # staight ribs
    rip_4_left = shape_creator.brick(center_lr=body_lr_center - rip_len_lr / 2 - vertebra_len_lr / 2,
                                     center_ap=body_ap_center,
                                     center_is=t4_is_center, len_lr=rip_len_lr, len_ap=rip_height, len_is=rip_height)
    rip_4_right = shape_creator.brick(center_lr=body_lr_center + rip_len_lr / 2 + vertebra_len_lr / 2,
                                      center_ap=body_ap_center,
                                      center_is=t4_is_center, len_lr=rip_len_lr, len_ap=rip_height, len_is=rip_height)
    sternum = shape_creator.brick(center_lr=body_lr_center, center_ap=body_ap_center + rip_len_ap,
                                  center_is=t2_is_center, len_lr=vertebra_len_lr, len_ap=2,
                                  len_is=t1_is_center - t3_is_center + vertebra_t_len_is)

    # Arms

    upper_arm_length = rng.integers(20 / 8 * scale_factor, 30 / 8 * scale_factor, size=1)[0]
    upper_arm_radius = rng.integers(2 / 8 * scale_factor, 4 / 8 * scale_factor, size=1)[0]
    lower_arm_length = rng.integers(20 / 8 * scale_factor, 30 / 8 * scale_factor, size=1)[0]
    lower_arm_radius = rng.integers(2 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[0]
    arm_lr_offset = shoulder_width + upper_arm_radius + 0.5
    arm_is_offset = rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[0]
    arm_upper_left = shape_creator.tube_vertical(center_lr=body_lr_center - arm_lr_offset,
                                                 center_ap=body_ap_center,
                                                 center_is=c1_is_center - upper_arm_length / 2 - clavicula_radius / 2 - arm_is_offset,
                                                 len_is=upper_arm_length, radius=upper_arm_radius)
    arm_upper_right = shape_creator.tube_vertical(center_lr=body_lr_center + arm_lr_offset,
                                                  center_ap=body_ap_center,
                                                  center_is=c1_is_center - upper_arm_length / 2 - clavicula_radius / 2 - arm_is_offset,
                                                  len_is=upper_arm_length, radius=upper_arm_radius)
    arm_lower_left = shape_creator.tube_vertical(center_lr=body_lr_center - arm_lr_offset,
                                                 center_ap=body_ap_center,
                                                 center_is=c1_is_center - lower_arm_length / 2 - upper_arm_length - clavicula_radius / 2 - arm_is_offset,
                                                 len_is=lower_arm_length, radius=lower_arm_radius)
    arm_lower_right = shape_creator.tube_vertical(center_lr=body_lr_center + arm_lr_offset,
                                                  center_ap=body_ap_center,
                                                  center_is=c1_is_center - lower_arm_length / 2 - upper_arm_length - clavicula_radius / 2 - arm_is_offset,
                                                  len_is=lower_arm_length, radius=lower_arm_radius)
    distance_hand_arm = rng.integers(1 / 8 * scale_factor, 3 / 8 * scale_factor, size=1)[0]
    carpal_bone_radius = 1 / 8 * scale_factor
    finger_distance_lr = 3 / 8 * scale_factor
    hand_left_carpal_1 = shape_creator.sphere(center_lr=body_lr_center - arm_lr_offset, center_ap=body_ap_center,
                                              center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm,
                                              radius=carpal_bone_radius)
    hand_left_carpal_2 = shape_creator.sphere(center_lr=body_lr_center - arm_lr_offset + finger_distance_lr,
                                              center_ap=body_ap_center,
                                              center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm,
                                              radius=carpal_bone_radius)
    hand_left_carpal_3 = shape_creator.sphere(center_lr=body_lr_center - arm_lr_offset - finger_distance_lr,
                                              center_ap=body_ap_center,
                                              center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm,
                                              radius=carpal_bone_radius)
    hand_right_carpal_1 = shape_creator.sphere(center_lr=body_lr_center + arm_lr_offset, center_ap=body_ap_center,
                                               center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm,
                                               radius=carpal_bone_radius)
    hand_right_carpal_2 = shape_creator.sphere(center_lr=body_lr_center + arm_lr_offset - finger_distance_lr,
                                               center_ap=body_ap_center,
                                               center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm,
                                               radius=carpal_bone_radius)
    hand_right_carpal_3 = shape_creator.sphere(center_lr=body_lr_center + arm_lr_offset + finger_distance_lr,
                                               center_ap=body_ap_center,
                                               center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm,
                                               radius=carpal_bone_radius)

    finger_length = rng.integers(4 / 8 * scale_factor, 5 / 8 * scale_factor, size=1)[0]
    finger_radius = 1 / 8 * scale_factor
    hand_left_finger_1 = shape_creator.tube_vertical(center_lr=body_lr_center - arm_lr_offset, center_ap=body_ap_center,
                                                     center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm - 2 * carpal_bone_radius - finger_length / 2,
                                                     len_is=finger_length, radius=finger_radius)
    hand_left_finger_2 = shape_creator.tube_vertical(center_lr=body_lr_center - arm_lr_offset + finger_distance_lr,
                                                     center_ap=body_ap_center,
                                                     center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm - 2 * carpal_bone_radius - finger_length / 2,
                                                     len_is=finger_length, radius=finger_radius)
    hand_left_finger_3 = shape_creator.tube_vertical(center_lr=body_lr_center - arm_lr_offset - finger_distance_lr,
                                                     center_ap=body_ap_center,
                                                     center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm - 2 * carpal_bone_radius - finger_length / 2,
                                                     len_is=finger_length, radius=finger_radius)
    hand_right_finger_1 = shape_creator.tube_vertical(center_lr=body_lr_center + arm_lr_offset,
                                                      center_ap=body_ap_center,
                                                      center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm - 2 * carpal_bone_radius - finger_length / 2,
                                                      len_is=finger_length, radius=finger_radius)
    hand_right_finger_2 = shape_creator.tube_vertical(center_lr=body_lr_center + arm_lr_offset - finger_distance_lr,
                                                      center_ap=body_ap_center,
                                                      center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm - 2 * carpal_bone_radius - finger_length / 2,
                                                      len_is=finger_length, radius=finger_radius)
    hand_right_finger_3 = shape_creator.tube_vertical(center_lr=body_lr_center + arm_lr_offset + finger_distance_lr,
                                                      center_ap=body_ap_center,
                                                      center_is=c1_is_center - lower_arm_length - upper_arm_length - clavicula_radius / 2 - arm_is_offset - distance_hand_arm - 2 * carpal_bone_radius - finger_length / 2,
                                                      len_is=finger_length, radius=finger_radius)
    # Hip

    cross_section_radius = rng.integers(3 / 8 * scale_factor, 5 / 8 * scale_factor, size=1)[0]
    torus_radius = shoulder_width * 3 / 4 - cross_section_radius / 2
    hip = shape_creator.torus(center_lr=body_lr_center, center_ap=body_ap_center, center_is=sacrum_is_center,
                              torus_radius=torus_radius, tube_cross_section_radius=cross_section_radius)
    # Legs

    femur_len = 30 / 8 * scale_factor
    femur_is_offet = rng.integers(1 / 8 * scale_factor, 4 / 8 * scale_factor, size=1)[0]
    femur_radius = rng.integers(3 / 8 * scale_factor, 5 / 8 * scale_factor, size=1)[0]
    femur_left = shape_creator.tube_vertical(center_lr=body_lr_center - torus_radius, center_ap=body_ap_center,
                                             center_is=sacrum_is_center - cross_section_radius - femur_is_offet - femur_len / 2,
                                             len_is=femur_len, radius=femur_radius)
    femur_right = shape_creator.tube_vertical(center_lr=body_lr_center + torus_radius, center_ap=body_ap_center,
                                              center_is=sacrum_is_center - cross_section_radius - femur_is_offet - femur_len / 2,
                                              len_is=femur_len, radius=femur_radius)

    data[skull_mask] = Labels.skull.value
    data[mandible_mask] = Labels.mandible.value
    data[c1_mask] = Labels.c1.value
    data[c2_mask] = Labels.c2.value
    data[t1_mask] = Labels.t1.value
    data[t2_mask] = Labels.t2.value
    data[t3_mask] = Labels.t3.value
    data[t4_mask] = Labels.t4.value
    data[l1_mask] = Labels.l1.value
    data[l2_mask] = Labels.l2.value
    data[rip_1_left] = Labels.rip_1_left.value
    data[rip_1_right] = Labels.rip_1_right.value
    data[rip_2_left] = Labels.rip_2_left.value
    data[rip_2_right] = Labels.rip_2_right.value
    data[rip_3_left] = Labels.rip_3_left.value
    data[rip_3_right] = Labels.rip_3_right.value
    data[rip_4_left] = Labels.rip_4_left.value
    data[rip_4_right] = Labels.rip_4_right.value
    data[arm_upper_left] = Labels.arm_upper_left.value
    data[arm_upper_right] = Labels.arm_upper_right.value
    data[arm_lower_left] = Labels.arm_lower_left.value
    data[arm_lower_right] = Labels.arm_lower_right.value
    data[clavicula_left_mask] = Labels.clavicula_left.value
    data[clavicula_right_mask] = Labels.clavicula_right.value
    data[sacrum_mask] = Labels.sacrum.value
    data[sternum] = Labels.sternum.value
    data[hip] = Labels.hip.value
    data[femur_left] = Labels.femur_left.value
    data[femur_right] = Labels.femur_right.value
    data[hand_left_carpal_1] = Labels.hand_left_carpal_1.value
    data[hand_left_carpal_2] = Labels.hand_left_carpal_2.value
    data[hand_left_carpal_3] = Labels.hand_left_carpal_3.value
    data[hand_left_finger_1] = Labels.hand_left_finger_1.value
    data[hand_left_finger_2] = Labels.hand_left_finger_2.value
    data[hand_left_finger_3] = Labels.hand_left_finger_3.value
    data[hand_right_carpal_1] = Labels.hand_right_carpal_1.value
    data[hand_right_carpal_2] = Labels.hand_right_carpal_2.value
    data[hand_right_carpal_3] = Labels.hand_right_carpal_3.value
    data[hand_right_finger_1] = Labels.hand_right_finger_1.value
    data[hand_right_finger_2] = Labels.hand_right_finger_2.value
    data[hand_right_finger_3] = Labels.hand_right_finger_3.value

    return data


def turn_segmentation_into_mock_ct(segmentation_data: np.ndarray, random_seed: int = 66) -> np.ndarray:
    """ Create synthetic CT from a ground truth label map.

    :param segmentation_data: scalar volume where every
    :param random_seed:
    :return: 3d volume of the same shape as segmentation_data
    """
    rng = np.random.default_rng(random_seed)
    random_dilation_rate = rng.integers(1, 4, size=1)[0]

    background_value = 0
    bone_hu_outer = 1000
    bone_hu_inner = 500
    soft_tissue_hu = 200

    body_parts = BodyPartLabelGroups().get_all_label_groups()

    background = background_value * np.ones_like(segmentation_data)

    bones = create_bones(segmentation_data, bone_hu_inner, bone_hu_outer)

    soft_tissue = create_soft_tissue(segmentation_data, body_parts, soft_tissue_hu, random_dilation_rate)

    mock_ct = np.maximum(background, np.maximum(bones, soft_tissue))
    mock_ct = mock_ct + create_noise(segmentation_data, rng)

    return mock_ct


def create_noise(segmentation_data, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=400, size=segmentation_data.shape)


def create_soft_tissue(segmentation_data: np.ndarray, body_parts: list, soft_tissue_hu: int,
                       random_dilation_rate: int) -> np.ndarray:
    """Create a convex hull around each body part, dilate, and fill with soft tissue HU intensity

    :param segmentation_data: 3D scalar volume where values >0 indicate bone, values=0 indicate background.
    :param body_parts: list of (list of bones that form one body part, e.g. arm)
    :param soft_tissue_hu:
    :param random_dilation_rate:
    :return: 3d volume of the same shape as segmentation_data
    """
    soft_tissue = np.zeros_like(segmentation_data)
    for body_part in body_parts:
        body_part_labels = [i.value for i in body_part]
        body_part_mask = np.isin(segmentation_data, body_part_labels)
        convex_hull_mask = skimage.morphology.convex_hull_image(body_part_mask)
        for _ in range(random_dilation_rate):
            convex_hull_mask = skimage.morphology.binary_dilation(convex_hull_mask)
        soft_tissue[convex_hull_mask] = soft_tissue_hu
    return soft_tissue


def create_bones(segmentation_data: np.ndarray, bone_hu_inner: int, bone_hu_outer: int) -> np.ndarray:
    """ Creates CT-like bones from a segmentation.

    :param segmentation_data: 3D scalar volume where values >0 indicate bone, values=0 indicate background.
    :param bone_hu_inner: HU intensity value for the inner part of the bones.
    :param bone_hu_outer: HU intensity value for the surface of the bones.
    :return: An array of the same shape as segmentation_data, containing bright voxels where the segmentation indicates
    the presence of bones.
    """
    bones = np.zeros_like(segmentation_data)
    for label in np.unique(segmentation_data):
        if label != 0:
            bone_mask = np.zeros_like(segmentation_data)
            bone_mask[segmentation_data == label] = label
            distance_map = scipy.ndimage.distance_transform_cdt(bone_mask, metric='chessboard', return_distances=True,
                                                                return_indices=False,
                                                                distances=None, indices=None)
            bones[distance_map > 0] = bone_hu_inner
            bones[distance_map == 1] = bone_hu_outer
    return bones


def create_mock_data(save_dir: Union[str, os.PathLike], num_scans: int, spatial_width: int = 64):
    """ Create num_scans number of synthetic ground truth segmentation and corresponding synthetic CT.

    :param save_dir: Where to save the results
    :param num_scans: How many scans to generate
    :param spatial_width: Width of result in pixels
    :return: None
    """
    for i in range(num_scans):
        print('create subject nr. {}'.format(i + 1))
        subject_id = i + 1
        subject = f"mock_{subject_id:03d}"
        subject_dir = os.path.join(save_dir, subject)
        if not os.path.isdir(subject_dir):
            os.makedirs(subject_dir)

        segmentation_data_array = create_segmentation_skeleton(size_lr=spatial_width, size_is=spatial_width * 2,
                                                               size_ap=spatial_width, random_seed=i)
        img_segmentation = nib.Nifti1Image(segmentation_data_array, np.eye(4))
        nib.save(img_segmentation, os.path.join(subject_dir, 'bones.nii.gz'))

        mock_ct_array = turn_segmentation_into_mock_ct(segmentation_data_array, random_seed=i)
        img_ct = nib.Nifti1Image(mock_ct_array, np.eye(4))
        nib.save(img_ct, os.path.join(subject_dir, 'volume.nii.gz'))


if __name__ == '__main__':
    create_mock_data(save_dir='synthetic_dataset_results', num_scans=50, spatial_width=128)
