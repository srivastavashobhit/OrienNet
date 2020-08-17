from rotator import  Rotator
from utils import angle_error
import mimetypes
import argparse

mimetypes.init()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("-u", "--url",
                        help="url",
                        required=True)
    args = parser.parse_args()

    path = args.url

    model_location = "models\oriennet-model.hdf5"
    model_error_func = angle_error
    rotator = Rotator(model_location, model_error_func)

    mimestart = mimetypes.guess_type(path)[0]

    if mimestart != None:
        mimestart = mimestart.split('/')[0]

    if mimestart == 'video':
        num_extraction = 50
        angle = rotator.get_video_rotation_angle(path, num_extraction)

    elif mimestart == 'image':
        angle = rotator.get_image_rotation_angle(path)

    else:
        print('Unsupported file type')

    print("Rotation angle -", angle)
