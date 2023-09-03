import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomCoarseDropout(A.CoarseDropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def apply_to_keypoints(
        self, keypoints, holes, **params):
        result = set(keypoints)
        return list(result)

keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)

augment = A.Compose([
    A.Perspective(p=0.5),
    A.ShiftScaleRotate (rotate_limit = 180, p=0.5),
    A.Resize(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomSunFlare(src_radius=50,p=0.5),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
    CustomCoarseDropout(p=1, min_height=8, min_width=8, max_height=30, max_width=30, max_holes=10),
    A.OneOf([
            A.RGBShift(p=0.5),
            A.ChannelShuffle(p=0.5),
            A.HueSaturationValue(p=0.6),
            A.RGBShift(p=0.5)
        ], p=0.5),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.15,0.2), contrast_limit=(-0.15,0.2)),
    A.AdvancedBlur(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], keypoint_params=keypoint_params)

normal_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], keypoint_params=keypoint_params)