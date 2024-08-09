import numpy as np

def cv_coord(from_rect=None, from_polar=None, from_cylin=None, from_sphere=None,
             to_rect=False, to_polar=False, to_cylin=False, to_sphere=False,
             degrees=False, double_in=False):
    """
    Convert coordinates between different systems (rectangular, polar, cylindrical, spherical).

    Parameters:
    - from_rect: Input coordinates in rectangular form (2D or 3D array).
    - from_polar: Input coordinates in polar form (2D array).
    - from_cylin: Input coordinates in cylindrical form (3D array).
    - from_sphere: Input coordinates in spherical form (3D array).
    - to_rect, to_polar, to_cylin, to_sphere: Set the desired output coordinate system.
    - degrees: Boolean to specify if angles should be in degrees.
    - double_in: Boolean to force double precision.

    Returns:
    - result: Converted coordinates in the desired output system.
    """
    def to_degrees(angle_rad):
        return angle_rad * (180.0 / np.pi)

    def to_radians(angle_deg):
        return angle_deg * (np.pi / 180.0)

    if degrees:
        ang_in = np.pi / 180.0
        ang_out = 180.0 / np.pi
    else:
        ang_in = 1.0
        ang_out = 1.0

    double = double_in or any(isinstance(arr, np.ndarray) and arr.dtype == np.float64 
                              for arr in [from_rect, from_polar, from_cylin, from_sphere])

    zero = np.zeros(1, dtype=np.float64) if double else np.zeros(1, dtype=np.float32)
    result = None

    if from_rect is not None and len(from_rect) > 1:
        if to_polar:
            rad = np.sqrt(from_rect[0]**2 + from_rect[1]**2)
            ang = np.where(rad != zero, ang_out * np.arctan2(from_rect[1], from_rect[0]), zero)
            result = np.array([ang, rad])
        
        elif to_cylin:
            rad = np.sqrt(from_rect[0]**2 + from_rect[1]**2)
            ang = np.where(rad != zero, ang_out * np.arctan2(from_rect[1], from_rect[0]), zero)
            if from_rect.shape[0] >= 3:
                result = np.array([ang, rad, from_rect[2]])
            else:
                result = np.array([ang, rad, np.zeros_like(ang)])
        
        elif to_sphere:
            rad = np.sqrt(from_rect[0]**2 + from_rect[1]**2 + (from_rect[2] if from_rect.shape[0] >= 3 else zero)**2)
            ang1 = np.where(rad != zero, ang_out * np.arctan2(from_rect[1], from_rect[0]), zero)
            ang2 = np.where(rad != zero, ang_out * np.arctan2(from_rect[2] if from_rect.shape[0] >= 3 else zero, 
                                                              np.sqrt(from_rect[0]**2 + from_rect[1]**2)), zero)
            result = np.array([ang1, ang2, rad])
        
        else:
            result = from_rect

    elif from_polar is not None and len(from_polar) > 1:
        if to_rect:
            result = np.array([from_polar[1] * np.cos(ang_in * from_polar[0]),
                               from_polar[1] * np.sin(ang_in * from_polar[0])])
        
        elif to_cylin:
            result = np.array([from_polar[0], from_polar[1], np.zeros_like(from_polar[0])])
        
        elif to_sphere:
            result = np.array([from_polar[0], np.zeros_like(from_polar[0]), from_polar[1]])
        
        else:
            result = from_polar

    elif from_cylin is not None and len(from_cylin) > 1:
        if to_rect:
            result = np.array([from_cylin[1] * np.cos(ang_in * from_cylin[0]),
                               from_cylin[1] * np.sin(ang_in * from_cylin[0]),
                               from_cylin[2]])
        
        elif to_polar:
            result = np.array([from_cylin[0], from_cylin[1]])
        
        elif to_sphere:
            rad = np.sqrt(from_cylin[1]**2 + from_cylin[2]**2)
            ang2 = np.where(rad != zero, ang_out * np.arctan2(from_cylin[2], from_cylin[1]), zero)
            result = np.array([from_cylin[0], ang2, rad])
        
        else:
            result = from_cylin

    elif from_sphere is not None and len(from_sphere) > 1:
        if to_rect:
            result = np.array([from_sphere[2] * np.cos(ang_in * from_sphere[0]) * np.cos(ang_in * from_sphere[1]),
                               from_sphere[2] * np.sin(ang_in * from_sphere[0]) * np.cos(ang_in * from_sphere[1]),
                               from_sphere[2] * np.sin(ang_in * from_sphere[1])])
        
        elif to_polar:
            result = np.array([from_sphere[0], from_sphere[2] * np.cos(ang_in * from_sphere[1])])
        
        elif to_cylin:
            result = np.array([from_sphere[0],
                               from_sphere[2] * np.cos(ang_in * from_sphere[1]),
                               from_sphere[2] * np.sin(ang_in * from_sphere[1])])
        
        else:
            result = from_sphere

    else:
        result = 0  # No valid input

    return result.astype(np.float64) if double else result.astype(np.float32)
