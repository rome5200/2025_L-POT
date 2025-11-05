import os
import pydicom
import numpy as np
import scipy.ndimage
from skimage import measure


class DICOMProcessor:
    """DICOM 파일 처리"""
    
    def __init__(self):
        pass
    
    def find_dicom_files(self, root):
        """DICOM 파일 찾기"""
        dcm_files = []
        for path, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_files.append(os.path.join(path, f))
        return sorted(dcm_files)
    
    def load_scan(self, input_folder):
        """DICOM 스캔 로드"""
        dcm_files = self.find_dicom_files(input_folder)
        if len(dcm_files) == 0:
            raise FileNotFoundError("DICOM 파일을 찾을 수 없습니다.")
        
        slices = [pydicom.dcmread(f) for f in dcm_files]
        
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except Exception:
            slices.sort(key=lambda x: float(x.SliceLocation))
        
        return slices
    
    def get_pixels_hu(self, slices):
        """HU 값으로 변환"""
        image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
        
        for i, s in enumerate(slices):
            intercept = s.RescaleIntercept
            slope = s.RescaleSlope
            
            if slope != 1:
                image[i] = (image[i].astype(np.float64) * slope).astype(np.int16)
            
            image[i] += np.int16(intercept)
        
        return image
    
    def resample(self, image, scan, new_spacing=[1, 1, 1]):
        """리샘플링"""
        spacing = np.array(
            [float(scan[0].SliceThickness)] + list(map(float, scan[0].PixelSpacing)),
            dtype=np.float32
        )
        
        resize_factor = spacing / new_spacing
        new_shape = np.round(image.shape * resize_factor)
        real_factor = new_shape / image.shape
        
        resampled = scipy.ndimage.zoom(image, real_factor, mode='nearest')
        
        return resampled, spacing
    
    def segment_lung_mask(self, image, fill_lung_structures=True):
        """폐 마스크 세그멘테이션"""
        binary = (image > -400).astype(np.int8) + 1
        labels = measure.label(binary)
        background = labels[0, 0, 0]
        binary[labels == background] = 2
        
        if fill_lung_structures:
            for i, sl in enumerate(binary):
                lab = measure.label(sl - 1)
                if lab.max() > 0:
                    lm = np.argmax(np.bincount(lab.flat)[1:]) + 1
                    sl[lab != lm] = 1
                binary[i] = sl + 1
        
        binary -= 1
        binary = 1 - binary
        labels = measure.label(binary, background=0)
        
        if labels.max() > 0:
            lm = np.argmax(np.bincount(labels.flat)[1:]) + 1
            binary[labels != lm] = 0
        
        return binary
    
    def load_and_process(self, input_folder):
        slices = self.load_scan(input_folder)
        image = self.get_pixels_hu(slices)
        resampled_image, spacing = self.resample(image, slices, new_spacing=[1, 1, 1])
        lung_mask = self.segment_lung_mask(resampled_image, fill_lung_structures=True)
        return {
            "image": resampled_image,
            "mask": lung_mask,
            "spacing": spacing,
            "slices": slices,
    }
