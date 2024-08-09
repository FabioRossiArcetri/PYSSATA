import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import rotate
import warnings
from scipy.interpolate import RegularGridInterpolator


# class Layer:
#     def __init__(self, size, pixel_pitch, A, phaseInNm, ongpu=False):
#         self.size = size
#         self.pixel_pitch = pixel_pitch
#         self.A = A
#         self.phaseInNm = phaseInNm
#         self.ongpu = ongpu

# class EF:
#     def __init__(self, size, pixel_pitch, ongpu=False):
#         self.size = size
#         self.pixel_pitch = pixel_pitch
#         self.ongpu = ongpu
#         self.A = np.zeros(size)
#         self.phaseInNm = np.zeros(size)
        
#     def product(self, layer, subrect=None):
#         if subrect:
#             x0, y0 = subrect
#             self.A = layer.A[x0:x0+self.size[0], y0:y0+self.size[1]]
#             self.phaseInNm = layer.phaseInNm[x0:x0+self.size[0], y0:y0+self.size[1]]
#         else:
#             self.A = layer.A
#             self.phaseInNm = layer.phaseInNm

#     def product_from_interpolated(self, layer, x_center, y_center, radius):
#         x = np.linspace(0, self.size[0] - 1, self.size[0])
#         y = np.linspace(0, self.size[1] - 1, self.size[1])
#         xx, yy = np.meshgrid(x, y)
#         interp_func = interp2d(x, y, layer.A, kind='linear')
#         self.A = interp_func(x_center + xx - radius, y_center + yy - radius)
#         interp_func = interp2d(x, y, layer.phaseInNm, kind='linear')
#         self.phaseInNm = interp_func(x_center + xx - radius, y_center + yy - radius)
    
#     def physical_prop(self, wavelengthInNm, propagator, temp_array=None):
#         # Placeholder for physical propagation implementation
#         pass

def single_layer2pupil_ef(layer_ef, polar_coordinate, height_source, update_ef=None, shiftXY=None, rotAnglePhInDeg=None,
                          magnify=None, pupil_position=None, temp_ef=None, doFresnel=False, propagator=None,
                          wavelengthInNm=None, temp_array=None):
    
    height_layer = layer_ef.size[0]
    pixel_pitch = layer_ef.pixel_pitch
    pixel_pupil = update_ef.size[0]

    diff_height = height_source - height_layer

    if (height_layer == 0 or (np.isfinite(height_source) and polar_coordinate[0] == 0)) and \
       (shiftXY is None) and (pupil_position is None) and (rotAnglePhInDeg is None) and (magnify is None):

        s_layer = layer_ef.size

        topleft = [(s_layer[0] - pixel_pupil) // 2, (s_layer[1] - pixel_pupil) // 2]
        update_ef.product(layer_ef, subrect=topleft)

    elif diff_height > 0:
        sec2rad = 4.848e-6
        degree2rad = np.pi / 180.
        r_angle = polar_coordinate[0] * sec2rad
        phi_angle = polar_coordinate[1] * degree2rad

        pixel_layer = layer_ef.size[0]
        half_pixel_layer_x = (pixel_layer - 1) / 2.
        half_pixel_layer_y = (pixel_layer - 1) / 2.
        if shiftXY is not None:
            half_pixel_layer_x -= shiftXY[0]
            half_pixel_layer_y -= shiftXY[1]

        if pupil_position is not None and pixel_layer > pixel_pupil and np.isfinite(height_source):
            pixel_position = r_angle * height_layer / pixel_pitch
            pixel_position_x = pixel_position * np.cos(phi_angle) + pupil_position[0] / pixel_pitch
            pixel_position_y = pixel_position * np.sin(phi_angle) + pupil_position[1] / pixel_pitch
        elif pupil_position is not None and pixel_layer > pixel_pupil and not np.isfinite(height_source):
            pixel_position = r_angle * height_source / pixel_pitch
            sky_pixel_position_x = pixel_position * np.cos(phi_angle)
            sky_pixel_position_y = pixel_position * np.sin(phi_angle)

            pupil_pixel_position_x = pupil_position[0] / pixel_pitch
            pupil_pixel_position_y = pupil_position[1] / pixel_pitch

            pixel_position_x = (sky_pixel_position_x - pupil_pixel_position_x) * height_layer / height_source + pupil_pixel_position_x
            pixel_position_y = (sky_pixel_position_y - pupil_pixel_position_y) * height_layer / height_source + pupil_pixel_position_y
        else:
            pixel_position = r_angle * height_layer / pixel_pitch
            pixel_position_x = pixel_position * np.cos(phi_angle)
            pixel_position_y = pixel_position * np.sin(phi_angle)

        if np.isfinite(height_source):
            pixel_pupmeta = pixel_pupil
        else:
            cone_coeff = abs(height_source - abs(height_layer)) / height_source
            pixel_pupmeta = pixel_pupil * cone_coeff

        if magnify is not None:
            pixel_pupmeta /= magnify
            tempA = layer_ef.A
            tempP = layer_ef.phaseInNm
            tempP[tempA == 0] = np.mean(tempP[tempA != 0])
            layer_ef.phaseInNm = tempP

        xx, yy = np.meshgrid(np.arange(pixel_pupil), np.arange(pixel_pupil))

        if rotAnglePhInDeg is not None:
            angle = (-rotAnglePhInDeg % 360) * np.pi / 180
            x = np.cos(angle) * xx - np.sin(angle) * yy + half_pixel_layer_x + pixel_position_x
            y = np.sin(angle) * xx + np.cos(angle) * yy + half_pixel_layer_y + pixel_position_y
            GRID = 0
        else:
            x = xx + half_pixel_layer_x + pixel_position_x
            y = yy + half_pixel_layer_y + pixel_position_y
            GRID = 1

        points = np.vstack((x.ravel(), y.ravel())).T
        interpolator_A = RegularGridInterpolator((np.arange(layer_ef.size[0]), np.arange(layer_ef.size[1])), layer_ef.A, bounds_error=False, fill_value=0)
        interpolator_phase = RegularGridInterpolator((np.arange(layer_ef.size[0]), np.arange(layer_ef.size[1])), layer_ef.phaseInNm, bounds_error=False, fill_value=0)
        pupil_ampl_temp = interpolator_A(points).reshape(pixel_pupil, pixel_pupil)
        pupil_phase_temp = interpolator_phase(points).reshape(pixel_pupil, pixel_pupil)


        update_ef.A *= pupil_ampl_temp
        update_ef.phaseInNm += pupil_phase_temp

    if doFresnel:
        update_ef.physical_prop(wavelengthInNm, propagator, temp_array=temp_array)

def layers2pupil_ef(layers, height_source, polar_coordinate_source, update_ef=None, shiftXY_list=None, rotAnglePhInDeg_list=None,
                    magnify_list=None, pupil_position=None, temp_ef=None, doFresnel=False, propagators=None,
                    wavelengthInNm=None, temp_array=None):

    if pupil_position is not None:
        warnings.warn('WARNING: pupil_position is not null')

    for i, layer in enumerate(layers):
        shiftXY = shiftXY_list[i] if shiftXY_list is not None and len(shiftXY_list) > 0 else None
        rotAnglePhInDeg = rotAnglePhInDeg_list[i] if rotAnglePhInDeg_list is not None and len(rotAnglePhInDeg_list) > 0 else None
        magnify = magnify_list[i] if magnify_list is not None and len(magnify_list) > 0 else None
        propagator = propagators[i] if propagators is not None else None
        
        single_layer2pupil_ef(layer, polar_coordinate_source, height_source, update_ef=update_ef, shiftXY=shiftXY,
                              rotAnglePhInDeg=rotAnglePhInDeg, magnify=magnify, pupil_position=pupil_position,
                              temp_ef=temp_ef, doFresnel=doFresnel, propagator=propagator,
                              wavelengthInNm=wavelengthInNm, temp_array=temp_array)
