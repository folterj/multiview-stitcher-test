import cv2 as cv
import numpy as np

from src.image.util import *


class Video:
    def __init__(self, filename, fps=1, scale=1, size=None, normalise=False):
        self.filename = filename
        self.fps = fps
        self.scale = scale
        self.size = size
        self.normalise = normalise
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
            if self.normalise:
                #self.min = np.mean(frame)
                #self.range = np.std(frame)
                self.min = np.quantile(frame, 0.01)
                self.range = np.quantile(frame, 0.99) - self.min
        frame = image_reshape(frame, self.size)
        if self.normalise:
            frame = float2int_image(np.clip((frame - self.min) / self.range, 0, 1))
        self.vidwriter.write(np.asarray(frame))

    def close(self):
        if self.vidwriter is not None:
            self.vidwriter.release()
