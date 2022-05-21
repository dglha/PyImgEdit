"""OpenCV Library"""
import cv2
import numpy as np
from PIL.Image import Image


class EffectFilter:
    """
    Class provide default CV2 colormap for apply to image filter
    """

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image

    @staticmethod
    def pink_dream(image: Image):
        """
        Apply COLORMAP_PINK
        :return: numpy array
        """
        # Apply pink colormap filter
        image = np.array(image)
        filtered_image = cv2.applyColorMap(image, cv2.COLORMAP_PINK)

        # Apply stylization filter that produces
        # image look like painted using water color
        filtered_image = cv2.stylization(
            filtered_image, sigma_s=60, sigma_r=0.6
        )
        return filtered_image

    @staticmethod
    def cyperpunk_2077(image: Image):
        """
        Apply COLORMAP_PLASMA
        :return: numpy array
        """
        image = np.array(image)
        filtered_image = cv2.applyColorMap(image, cv2.COLORMAP_PLASMA)
        # Apply Edge Preserving Filter (Bộ lọc làm mờ cạnh)
        # flags = 1 Use RECURS_FILTER
        # that 3.5x faster than 2 = NORMCONV_FILTER
        filtered_image = cv2.edgePreservingFilter(
            image, flags=1, sigma_r=0.6, sigma_s=40
        )
        return filtered_image

    @staticmethod
    def snowy(image: Image):
        """
        Apply BRG2GRAY effect
        :return: numpy array
        """
        image = np.array(image)
        # First, convert to grayscale image
        snowy_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        snowy_image_blur = cv2.GaussianBlur(
            snowy_image, (25, 25), 0
        )  # (25, 25) Kernel size
        return cv2.divide(snowy_image, snowy_image_blur, scale=250.0)

    @staticmethod
    def pastel(image: Image):
        """
        Apply COLORMAP_JET
        :return: numpy array
        """
        image = np.array(image)
        pastel_image = cv2.medianBlur(image, 5)
        return cv2.applyColorMap(pastel_image, cv2.COLORMAP_JET)

    @staticmethod
    def firestorm(image: Image):
        """
        Apply negative COLORMAP_PARULA
        :return: numpy array
        """
        image = np.array(image)
        firestorm_image = cv2.applyColorMap(image, cv2.COLORMAP_PARULA)
        return cv2.bitwise_not(firestorm_image)

    @staticmethod
    def ice(image: Image):
        """
        Apply COLORMAP_OCEAN
        :return: numpy array
        """
        image = np.array(image)
        ice_image = cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)
        # Apply Edge Preserving Filter (Bộ lọc làm mờ cạnh)
        # flags = 1 Use RECURS_FILTER that 3.5x faster than 2 = NORMCONV_FILTER
        # ice_image = cv2.edgePreservingFilter(
        #     ice_image, flags=1, sigma_r=0.5, sigma_s=70
        # )
        return ice_image

    @staticmethod
    def darkness(image: Image):
        """
        Make image darker
        :return: numpy array
        """
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_blur = cv2.GaussianBlur(gray_image, (25, 25), 0)
        darkness_image = cv2.divide(gray_image, gray_image_blur, scale=250.0)

        return cv2.bitwise_not(darkness_image)

    @staticmethod
    def gray_nostalgia(image: Image):
        """
        Apply COLORMAP_BONE
        :return: numpy array
        """
        image = np.array(image)
        mask_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_image = cv2.medianBlur(mask_image, 3)
        nostalgia_image = cv2.applyColorMap(mask_image, cv2.COLORMAP_BONE)

        return nostalgia_image

    @staticmethod
    def sweet_dream(image: Image):
        image = np.array(image)
        sweet_image = cv2.applyColorMap(image, cv2.COLORMAP_TWILIGHT_SHIFTED)
        sweet_image = cv2.edgePreservingFilter(
            sweet_image, flags=1, sigma_r=0.6, sigma_s=40
        )

        return sweet_image

    @staticmethod
    def cartoon(image: Image):
        image = np.array(image)
        color = cv2.bilateralFilter(image, d=9, sigmaColor=200, sigmaSpace=200)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5
        )
        new_image = cv2.bitwise_and(color, color, mask=edges)
        return new_image
