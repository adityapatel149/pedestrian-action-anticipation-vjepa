 
def _make_transforms(crop_size=256):
    from ..video_classification_frozen.utils import make_transforms

    return make_transforms(crop_size=crop_size, training=False)


def vjepa2_preprocessor(*, pretrained: bool = True, **kwargs):
    crop_size = kwargs.get("crop_size", 256)
    return _make_transforms(crop_size=crop_size)
