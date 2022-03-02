#  Copyright (c) 2022. Eva Schnider

import enum
from dataclasses import dataclass


class Labels(enum.Enum):
    skull = 1
    mandible = 2
    c1 = 3
    c2 = 4
    t1 = 5
    t2 = 6
    t3 = 7
    t4 = 8
    rip_1_left = 9
    rip_1_right = 10
    rip_2_left = 11
    rip_2_right = 12
    rip_3_left = 13
    rip_3_right = 14
    rip_4_left = 15
    rip_4_right = 16
    arm_upper_left = 17
    arm_upper_right = 18
    arm_lower_left = 19
    arm_lower_right = 20
    clavicula_left = 21
    clavicula_right = 22
    l1 = 23
    l2 = 24
    sacrum = 25
    sternum = 26
    hip = 27
    femur_left = 28
    femur_right = 29
    hand_left_carpal_1 = 30
    hand_left_carpal_2 = 31
    hand_left_carpal_3 = 32
    hand_left_finger_1 = 33
    hand_left_finger_2 = 34
    hand_left_finger_3 = 35
    hand_right_carpal_1 = 36
    hand_right_carpal_2 = 37
    hand_right_carpal_3 = 38
    hand_right_finger_1 = 39
    hand_right_finger_2 = 40
    hand_right_finger_3 = 41


@dataclass
class BodyPartLabelGroups:
    torso_upper_labels = [Labels.c1, Labels.c2, Labels.t1, Labels.t2, Labels.t3, Labels.t4, Labels.l1,
                          Labels.rip_1_left, Labels.rip_2_left, Labels.rip_3_left, Labels.rip_1_right,
                          Labels.rip_2_right, Labels.rip_3_right, Labels.rip_4_left, Labels.rip_4_right,
                          Labels.sternum, Labels.clavicula_left, Labels.clavicula_right]
    torso_lower_labels = [Labels.l1, Labels.l2, Labels.sacrum, Labels.hip]
    arm_left_labels = [Labels.arm_lower_left, Labels.arm_upper_left, Labels.hand_left_finger_1,
                       Labels.hand_left_finger_2, Labels.hand_left_finger_3, Labels.hand_left_carpal_1,
                       Labels.hand_left_carpal_2, Labels.hand_left_carpal_3]
    arm_right_labels = [Labels.arm_lower_right, Labels.arm_upper_right, Labels.hand_right_finger_1,
                        Labels.hand_right_finger_2, Labels.hand_right_finger_3, Labels.hand_right_carpal_1,
                        Labels.hand_right_carpal_2, Labels.hand_right_carpal_3]
    head_neck_labels = [Labels.skull, Labels.mandible, Labels.c1]
    leg_left_labels = [Labels.femur_left]
    leg_right_labels = [Labels.femur_right]

    def get_all_label_groups(self):
        return [self.torso_upper_labels,
                self.torso_lower_labels,
                self.arm_left_labels,
                self.arm_right_labels,
                self.head_neck_labels,
                self.leg_right_labels,
                self.leg_left_labels]
