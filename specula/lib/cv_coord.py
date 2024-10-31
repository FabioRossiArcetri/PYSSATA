
# Translation of IDL cv_coord.pro

import numpy as np

from specula import xp
from specula import float_dtype

def cv_coord(from_rect=None, from_polar=None, from_cylin=None, from_sphere=None,
             to_rect=False, to_polar=False, to_cylin=False, to_sphere=False,
             degrees=False, double=False):
    
    def to_double(arr):
        return arr.astype(xp.float64) if double else arr.astype(xp.float32)
    
    if from_rect is not None:
        from_rect = xp.array(from_rect, dtype=float_dtype)
        if double:
            from_rect = to_double(from_rect)
        zero = 0.0 if not double else 0.0
        ang_out = xp.pi/180.0 if degrees else 1.0
        
        if to_polar:
            rad = xp.sqrt(from_rect[0]**2 + from_rect[1]**2)
            ang = xp.arctan2(from_rect[1], from_rect[0]) * ang_out
            return xp.array([ang, rad], dtype=float_dtype)
        
        if to_cylin:
            rad = xp.sqrt(from_rect[0]**2 + from_rect[1]**2)
            ang = xp.arctan2(from_rect[1], from_rect[0]) * ang_out
            z = from_rect[2] if from_rect.shape[0] > 2 else zero
            return xp.array([ang, rad, z], dtype=float_dtype)
        
        if to_sphere:
            rad = xp.sqrt(from_rect[0]**2 + from_rect[1]**2 + from_rect[2]**2)
            ang1 = xp.arctan2(from_rect[1], from_rect[0]) * ang_out
            ang2 = xp.arctan2(from_rect[2], xp.sqrt(from_rect[0]**2 + from_rect[1]**2)) * ang_out
            return xp.array([ang1, ang2, rad], dtype=float_dtype)
        
        return from_rect
    
    elif from_polar is not None:
        from_polar = xp.array(from_polar, dtype=float_dtype)
        if double:
            from_polar = to_double(from_polar)
        ang_in = xp.pi/180.0 if degrees else 1.0
        
        if to_rect:
            x = from_polar[1] * xp.cos(from_polar[0] * ang_in)
            y = from_polar[1] * xp.sin(from_polar[0] * ang_in)
            return xp.array([x, y], dtype=float_dtype)
        
        if to_cylin:
            z = zero
            return xp.array([from_polar[0], from_polar[1], z], dtype=float_dtype)
        
        if to_sphere:
            return xp.array([from_polar[0], zero, from_polar[1]], dtype=float_dtype)
        
        return from_polar
    
    elif from_cylin is not None:
        from_cylin = xp.array(from_cylin, dtype=float_dtype)
        if double:
            from_cylin = to_double(from_cylin)
        ang_in = xp.pi/180.0 if degrees else 1.0
        ang_out = 180.0/xp.pi if degrees else 1.0
        
        if to_rect:
            x = from_cylin[1] * xp.cos(from_cylin[0] * ang_in)
            y = from_cylin[1] * xp.sin(from_cylin[0] * ang_in)
            z = from_cylin[2]
            return xp.array([x, y, z], dtype=float_dtype)
        
        if to_polar:
            return xp.array([from_cylin[0], from_cylin[1]], dtype=float_dtype)
        
        if to_sphere:
            rad = xp.sqrt(from_cylin[1]**2 + from_cylin[2]**2)
            ang1 = from_cylin[0]
            ang2 = xp.arctan2(from_cylin[2], from_cylin[1]) * ang_out
            return xp.array([ang1, ang2, rad], dtype=float_dtype)
        
        return from_cylin
    
    elif from_sphere is not None:
        from_sphere = xp.array(from_sphere, dtype=float_dtype)
        if double:
            from_sphere = to_double(from_sphere)
        ang_in = xp.pi/180.0 if degrees else 1.0
        
        if to_rect:
            x = from_sphere[2] * xp.cos(from_sphere[0] * ang_in) * xp.cos(from_sphere[1] * ang_in)
            y = from_sphere[2] * xp.sin(from_sphere[0] * ang_in) * xp.cos(from_sphere[1] * ang_in)
            z = from_sphere[2] * xp.sin(from_sphere[1] * ang_in)
            return xp.array([x, y, z], dtype=float_dtype)
        
        if to_polar:
            rad = from_sphere[2] * xp.cos(from_sphere[1] * ang_in)
            return xp.array([from_sphere[0], rad], dtype=float_dtype)
        
        if to_cylin:
            rad = from_sphere[2] * xp.cos(from_sphere[1] * ang_in)
            z = from_sphere[2] * xp.sin(from_sphere[1] * ang_in)
            return xp.array([from_sphere[0], rad, z], dtype=float_dtype)
        
        return from_sphere
    
    return 0  # If no valid input is given

