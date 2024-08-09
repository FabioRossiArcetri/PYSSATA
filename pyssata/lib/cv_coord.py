
# Translation of IDL cv_coord.pro

import numpy as np

def cv_coord(from_rect=None, from_polar=None, from_cylin=None, from_sphere=None,
             to_rect=False, to_polar=False, to_cylin=False, to_sphere=False,
             degrees=False, double=False):
    
    def to_double(arr):
        return arr.astype(np.float64) if double else arr.astype(np.float32)
    
    if from_rect is not None:
        from_rect = np.array(from_rect)
        if double:
            from_rect = to_double(from_rect)
        zero = 0.0 if not double else 0.0
        ang_out = np.pi/180.0 if degrees else 1.0
        
        if to_polar:
            rad = np.sqrt(from_rect[0]**2 + from_rect[1]**2)
            ang = np.arctan2(from_rect[1], from_rect[0]) * ang_out
            return np.array([ang, rad])
        
        if to_cylin:
            rad = np.sqrt(from_rect[0]**2 + from_rect[1]**2)
            ang = np.arctan2(from_rect[1], from_rect[0]) * ang_out
            z = from_rect[2] if from_rect.shape[0] > 2 else zero
            return np.array([ang, rad, z])
        
        if to_sphere:
            rad = np.sqrt(from_rect[0]**2 + from_rect[1]**2 + from_rect[2]**2)
            ang1 = np.arctan2(from_rect[1], from_rect[0]) * ang_out
            ang2 = np.arctan2(from_rect[2], np.sqrt(from_rect[0]**2 + from_rect[1]**2)) * ang_out
            return np.array([ang1, ang2, rad])
        
        return from_rect
    
    elif from_polar is not None:
        from_polar = np.array(from_polar)
        if double:
            from_polar = to_double(from_polar)
        ang_in = np.pi/180.0 if degrees else 1.0
        
        if to_rect:
            x = from_polar[1] * np.cos(from_polar[0] * ang_in)
            y = from_polar[1] * np.sin(from_polar[0] * ang_in)
            return np.array([x, y])
        
        if to_cylin:
            z = zero
            return np.array([from_polar[0], from_polar[1], z])
        
        if to_sphere:
            return np.array([from_polar[0], zero, from_polar[1]])
        
        return from_polar
    
    elif from_cylin is not None:
        from_cylin = np.array(from_cylin)
        if double:
            from_cylin = to_double(from_cylin)
        ang_in = np.pi/180.0 if degrees else 1.0
        ang_out = 180.0/np.pi if degrees else 1.0
        
        if to_rect:
            x = from_cylin[1] * np.cos(from_cylin[0] * ang_in)
            y = from_cylin[1] * np.sin(from_cylin[0] * ang_in)
            z = from_cylin[2]
            return np.array([x, y, z])
        
        if to_polar:
            return np.array([from_cylin[0], from_cylin[1]])
        
        if to_sphere:
            rad = np.sqrt(from_cylin[1]**2 + from_cylin[2]**2)
            ang1 = from_cylin[0]
            ang2 = np.arctan2(from_cylin[2], from_cylin[1]) * ang_out
            return np.array([ang1, ang2, rad])
        
        return from_cylin
    
    elif from_sphere is not None:
        from_sphere = np.array(from_sphere)
        if double:
            from_sphere = to_double(from_sphere)
        ang_in = np.pi/180.0 if degrees else 1.0
        
        if to_rect:
            x = from_sphere[2] * np.cos(from_sphere[0] * ang_in) * np.cos(from_sphere[1] * ang_in)
            y = from_sphere[2] * np.sin(from_sphere[0] * ang_in) * np.cos(from_sphere[1] * ang_in)
            z = from_sphere[2] * np.sin(from_sphere[1] * ang_in)
            return np.array([x, y, z])
        
        if to_polar:
            rad = from_sphere[2] * np.cos(from_sphere[1] * ang_in)
            return np.array([from_sphere[0], rad])
        
        if to_cylin:
            rad = from_sphere[2] * np.cos(from_sphere[1] * ang_in)
            z = from_sphere[2] * np.sin(from_sphere[1] * ang_in)
            return np.array([from_sphere[0], rad, z])
        
        return from_sphere
    
    return 0  # If no valid input is given

