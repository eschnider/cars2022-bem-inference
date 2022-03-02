#  Copyright (c) 2022. Eva Schnider

import numpy as np


class ShapeCreator:
    def __init__(self, canvas_size_lr, canvas_size_is, canvas_size_ap):
        self.canvas_size_ap = canvas_size_ap
        self.canvas_size_is = canvas_size_is
        self.canvas_size_lr = canvas_size_lr
        # create a meshgrid with coordinates in all three dimentions
        self.xx, self.yy, self.zz = np.mgrid[:canvas_size_lr, :canvas_size_ap, :canvas_size_is]

    def sphere(self, center_lr, center_ap, center_is, radius):
        # sphere contains the squared distance to the center point
        sphere_equation = (self.xx - center_lr) ** 2 + \
                          (self.yy - center_ap) ** 2 + \
                          (self.zz - center_is) ** 2 - radius ** 2
        sphere_mask = sphere_equation < 0
        return sphere_mask

    def half_sphere_lower(self, center_lr, center_ap, center_is, radius):
        sphere_mask = self.sphere(center_lr=center_lr, center_ap=center_ap, center_is=center_is, radius=radius)
        half_sphere_mask = np.logical_and(sphere_mask, self.zz < center_is)
        return half_sphere_mask

    def half_sphere_upper(self, center_lr, center_ap, center_is, radius):
        sphere_mask = self.sphere(center_lr=center_lr, center_ap=center_ap, center_is=center_is, radius=radius)
        half_sphere_mask = np.logical_and(sphere_mask, self.zz > center_is)
        return half_sphere_mask

    def brick(self, center_lr, center_ap, center_is, len_lr, len_ap, len_is):
        slice_lr = np.logical_and(center_lr - len_lr / 2 < self.xx, self.xx <= center_lr + len_lr / 2)
        slice_ap = np.logical_and(center_ap - len_ap / 2 < self.yy, self.yy <= center_ap + len_ap / 2)
        slice_is = np.logical_and(center_is - len_is / 2 < self.zz, self.zz <= center_is + len_is / 2)
        brick_mask = np.logical_and(slice_lr, np.logical_and(slice_ap, slice_is))
        return brick_mask

    def pyramid_on_tip(self, center_base_lr, center_base_ap, center_base_is, base_len, height):
        slope = height / base_len
        lr_equation = -slope * np.abs(self.xx - center_base_lr) + height >= np.abs(self.zz - center_base_is)
        ap_equation = -slope * np.abs(self.yy - center_base_ap) + height >= np.abs(self.zz - center_base_is)
        pyramid_mask = np.logical_and(self.zz <= center_base_is, np.logical_and(lr_equation, ap_equation))
        return pyramid_mask

    def tube_vertical(self, center_lr, center_ap, center_is, len_is, radius):
        # circle in x and y dimension contains the squared distance to the center point
        sphere_equation = (self.xx - center_lr) ** 2 + \
                          (self.yy - center_ap) ** 2 - radius ** 2
        tube_mask = np.logical_and(sphere_equation < 0,
                                   np.logical_and(center_is - len_is / 2 < self.zz, self.zz <= center_is + len_is / 2))
        return tube_mask

    def tube_horizontal(self, center_lr, center_ap, center_is, len_lr, radius):
        # circle in x and y dimension contains the squared distance to the center point
        sphere_equation = (self.yy - center_ap) ** 2 + \
                          (self.zz - center_is) ** 2 - radius ** 2
        tube_mask = np.logical_and(sphere_equation < 0,
                                   np.logical_and(center_lr - len_lr / 2 < self.xx, self.xx <= center_lr + len_lr / 2))
        return tube_mask

    def u_horizontal_open_right(self, center_lr, center_ap, center_is, breadth, height, thickness):
        """ Create a U shape that lies on the horizontal plane, opening towards the right.

        :param center_lr: left/right center on the U shape, i.e. were the . is placed in |_._|
        :param center_ap: anterior/posterior center on the U shape, i.e. were the . is placed in |_._|
        :param center_is: inferior/superior center on the U shape, i.e. were the . is placed in |_._|
        :param breadth: The distance between the poles | and | in |__|.
        :param height: The length of the pole | in |__|.
        :param thickness: If you turn the U from 2D to 3D, how much depth you add.
        """

        pole_p = self.brick(center_lr=center_lr + height / 2 - thickness / 2, center_ap=center_ap - breadth / 2,
                            center_is=center_is, len_lr=height, len_ap=thickness, len_is=thickness)
        pole_a = self.brick(center_lr=center_lr + height / 2 - thickness / 2, center_ap=center_ap + breadth / 2,
                            center_is=center_is, len_lr=height, len_ap=thickness, len_is=thickness)
        bar = self.brick(center_lr=center_lr, center_ap=center_ap, center_is=center_is, len_lr=thickness,
                         len_ap=breadth, len_is=thickness)
        u_mask = np.logical_or(bar, np.logical_or(pole_a, pole_p))
        return u_mask

    def u_horizontal_open_left(self, center_lr, center_ap, center_is, breadth, height, thickness):
        """ Create a U shape that lies on the horizontal plane, opening towards the left.

        :param center_lr: left/right center on the U shape, i.e. were the . is placed in |_._|
        :param center_ap: anterior/posterior center on the U shape, i.e. were the . is placed in |_._|
        :param center_is: inferior/superior center on the U shape, i.e. were the . is placed in |_._|
        :param breadth: The distance between the poles | and | in |__|.
        :param height: The length of the pole | in |__|.
        :param thickness: If you turn the U from 2D to 3D, how much depth you add.
        """

        pole_p = self.brick(center_lr=center_lr - height / 2 + thickness / 2, center_ap=center_ap - breadth / 2,
                            center_is=center_is, len_lr=height, len_ap=thickness, len_is=thickness)
        pole_a = self.brick(center_lr=center_lr - height / 2 + thickness / 2, center_ap=center_ap + breadth / 2,
                            center_is=center_is, len_lr=height, len_ap=thickness, len_is=thickness)
        bar = self.brick(center_lr=center_lr, center_ap=center_ap, center_is=center_is, len_lr=thickness,
                         len_ap=breadth, len_is=thickness)
        u_mask = np.logical_or(bar, np.logical_or(pole_a, pole_p))
        return u_mask

    def torus(self, center_lr, center_ap, center_is, torus_radius, tube_cross_section_radius):
        equation = np.square(np.sqrt((self.xx - center_lr) ** 2 + (self.yy - center_ap) ** 2) - torus_radius) + (
                    self.zz - center_is) ** 2 < tube_cross_section_radius ** 2
        return equation
