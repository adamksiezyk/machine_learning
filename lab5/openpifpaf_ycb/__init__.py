import openpifpaf
from .ycb_video_kp import YCBVideoKp


def register():
    openpifpaf.DATAMODULES['ycb_video'] = YCBVideoKp