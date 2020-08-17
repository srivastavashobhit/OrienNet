import cv2

from keras.models import load_model
from keras.optimizers import SGD

from utils import get_frames_from_videos, cal_rotation_angle


class Rotator(object):

    def __init__(self, model_location, model_error_func):

        self.model_location = model_location
        self.error = model_error_func
        self.model = None
        self.build_model()

    def build_model(self):

        self.model = load_model(self.model_location, custom_objects={'angle_error': self.error})
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.01, momentum=0.9),
                      metrics=[self.error])

    def get_image_rotation_angle(self, image_path):
        dim = (224, 224)
        image = cv2.imread(image_path)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        angles_array = self.model.predict(image)
        rotation_angle = cal_rotation_angle(angles_array)
        return rotation_angle

    def get_video_rotation_angle(self, video_path, num_extraction):

        input = get_frames_from_videos(video_path, num_extraction)
        angles_array = self.model.predict_on_batch(input)
        rotation_angle = cal_rotation_angle(angles_array)
        return rotation_angle

