import random
import numpy as np
import cv2

class DataAugmentor:  # inspired by https://github.com/MikaeldeVerdier/mahjong/blob/main/ssd/augment.py
    def __init__(self, image_size, **kwargs):
        self.expand = self.random_expand(**kwargs.get("expand", {}))
        self.crop = self.random_crop(**kwargs.get("crop", {}))
        self.horizontal_flip = self.random_flip(dim="horizontal", **kwargs.get("horizontal_flip", {}))
        # self.vertical_flip = self.random_flip(dim="vertical", **kwargs.get("vertical_flip", {}))
        self.rotate = self.random_rotate(**kwargs.get("rotate", {}))
        self.perspective_warp = self.random_perspective_warp(**kwargs.get("perspective_warp", {}))
        self.homography_transform = self.random_homography_transform(**kwargs.get("homography_transform", {}))
        self.motion_blur = self.random_motion_blur(**kwargs.get("motion_blur", {}))
        self.resize = self.random_resize(image_size[0], image_size[1], **kwargs.get("resize", {}))

        self.augmentations = [  # Only geometric augmentations, I think
            self.expand,
            self.crop,
            self.horizontal_flip,
            self.rotate,
            # self.perspective_warp,
            # self.homography_transform,
            self.motion_blur,
            self.resize
        ]

    def __call__(self, images):
        augmented_images = []
        for image in images:  # could try to vectorize all augmentations
            augmented_image = image
            for augmentation in self.augmentations:
                augmented_image = augmentation(augmented_image)

            augmented_images.append(augmented_image)

        return augmented_images

    def random_expand(self, lower_bound=1, upper_bound=1.5, background_color=[0, 0, 0], p=0.5):
        def transform(image):
            if np.random.rand() > p:
                return image

            height, width, depth = image.shape
            ratio = np.random.uniform(lower_bound, upper_bound)

            top = int(np.random.uniform(0, height * (ratio - 1)))
            left = int(np.random.uniform(0, width * (ratio - 1)))
            new_height = int(height * ratio)
            new_width = int(width * ratio)

            expanded_image = np.zeros((new_height, new_width, depth), dtype=image.dtype)
            expanded_image[:] = background_color
            expanded_image[top:(top + height), left:(left + width)] = image

            return expanded_image

        return transform


    def random_crop(self, min_scale=0.8, max_scale=1, min_ar=0.5, max_ar=2):
        def transform(image):
            height, width, _ = image.shape

            current_image = image.copy()

            new_ar = np.random.uniform(min_ar, max_ar)
            new_h = int(np.random.uniform(height * min_scale, height * max_scale))  # np.random.randint ?
            new_w = int(new_h * new_ar)

            top = max(int(np.random.uniform(0, height - new_h)), 0)
            left = max(int(np.random.uniform(0, width - new_w)), 0)

            rect = np.array([left, top, left + new_w, top + new_h])
            cropped_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]

            return cropped_image  # double break

        return transform


    def random_flip(self, dim="horizontal", p=0.5):
        def transform(image):
            if np.random.rand() > p:
                return image

            if dim == "horizontal":
                flipped_image = image[:, ::-1]
            elif dim == "vertical":
                flipped_image = image[::-1]
            else:
                flipped_image = image

            return flipped_image

        return transform

    def random_rotate(self, min_angle=0, max_angle=30, p=0.5):
        def transform(image):
            if np.random.rand() > p:
                return image

            angle = np.random.uniform(min_angle, max_angle)

            # pil_image = Image.fromarray(image)
            # rotated_image_pil = pil_image.rotate(angle, expand=True)
            # rotated_image = np.array(rotated_image_pil)

            height, width = image.shape[:2]

            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            cos_val = np.abs(rotation_matrix[0, 0])
            sin_val = np.abs(rotation_matrix[0, 1])

            new_width = int((height * sin_val) + (width * cos_val))
            new_height = int((height * cos_val) + (width * sin_val))

            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

            return rotated_image

        return transform

    def random_perspective_warp(self, min_factor=0, max_factor=0.2, p=0.5):
        def transform(image):
            if np.random.rand() > p:
                return image

            height, width = image.shape[:2]

            factor = np.random.uniform(min_factor, max_factor)

            src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            dst_points = src_points + np.random.uniform(-width * factor, width * factor, src_points.shape).astype(np.float32)

            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))

            return warped_image

        return transform

    def random_homography_transform(self, max_homography_factor=0.3, p=0.5):
        def transform(image):
            if np.random.rand() > p:
                return image

            height, width = image.shape[:2]

            src_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            dst_points = src_points + np.random.uniform(-width * max_homography_factor, width * max_homography_factor, src_points.shape).astype(np.float32)

            homography_matrix, _ = cv2.findHomography(src_points, dst_points)
            warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))

            return warped_image

        return transform

    def random_motion_blur(self, min_kernel_size=1, max_kernel_size=5, p=0.5):
        def transform(image):
            if np.random.rand() > p:
                return image

            kernel_size = int(np.random.uniform(min_kernel_size, max_kernel_size))

            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel /= kernel_size

            blurred_image = cv2.filter2D(image, -1, kernel)

            return blurred_image

        return transform

    def random_resize(self, width, height, p=1):
        interpolation_options = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

        def transform(image):
            if np.random.rand() > p:
                return image

            chosen_interpolation = random.choice(interpolation_options)
            resized_image = cv2.resize(image, (width, height), interpolation=chosen_interpolation)

            return resized_image

        return transform
