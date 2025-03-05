import cv2 as cv
import numpy as np

from src.image.util import *


class Video:
    def __init__(self, filename, fps=1, scale=1, size=None):
        self.filename = filename
        self.fps = fps
        self.scale = scale
        self.size = size
        self.min = 0
        self.range = 1
        self.vidwriter = None

    def write(self, frame):
        frame = frame.squeeze()
        if self.scale != 1:
            frame = cv.resize(np.asarray(frame), None, fx=self.scale, fy=self.scale)
        if self.vidwriter is None:
            if self.size is None:
                height, width = frame.shape[:2]
                self.size = (width, height)
            self.vidwriter = cv.VideoWriter(self.filename, -1, self.fps, self.size)
        frame = image_reshape(frame, self.size)
        self.vidwriter.write(np.asarray(frame))

    def close(self):
        if self.vidwriter is not None:
            self.vidwriter.release()
