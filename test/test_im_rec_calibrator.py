

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula.data_objects.slopes import Slopes
from specula.base_value import BaseValue
from specula.processing_objects.im_rec_calibrator import ImRecCalibrator

class TestImRecCalibrator(unittest.TestCase):

    def test_existing_im_file_is_detected(self):

        data_dir = '/tmp'
        im_filename = 'test_im.fits'
        im_path = os.path.join(data_dir, im_filename)
        open(im_path, 'a').close()
        
        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='x', im_tag='test_im')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        
        with self.assertRaises(FileExistsError):
            calibrator.setup(1, 1)
        
    def test_existing_rec_file_is_detected(self):

        data_dir = '/tmp'
        rec_filename = 'test_rec.fits'
        rec_path = os.path.join(data_dir, rec_filename)
        open(rec_path, 'a').close()
        
        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='test_rec')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        
        with self.assertRaises(FileExistsError):
            calibrator.setup(1, 1)

    def test_existing_im_file_is_not_detected_if_not_requested(self):

        data_dir = '/tmp'
        im_filename = 'test_im.fits'
        rec_filename = 'test_rec.fits'
        im_path = os.path.join(data_dir, im_filename)
        rec_path = os.path.join(data_dir, rec_filename)
        open(im_path, 'a').close()
        if os.path.exists(rec_path):
            os.unlink(rec_path)

        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImRecCalibrator(nmodes=10, data_dir=data_dir, rec_tag='test_rec')
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)

        # Does not raise        
        calibrator.setup(1, 1)
                
