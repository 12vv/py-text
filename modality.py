from enum import Enum

class Modality(Enum):
    IMAGE = "image"
    IMAGE_MASK = "image_mask"
    TEXT = "text"
    TEXT_MASK = "text_mask"
    VIDEO = "video"
    VIDEO_MASK = "video_mask"
    AUDIO = "audio"
    AUDIO_MASK = "audio_mask"
