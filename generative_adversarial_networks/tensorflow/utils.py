import requests

from scipy.misc import imread, imsave, imresize
from tqdm import tqdm


def scale_image(image):
    # scale to (-1, +1)
    return (image / 255.0) * 2 - 1


def crop_and_resave(input_file, output_dir):
    # naively crop the center, instead of finding the face location using e.g OpenCV
    image = imread(input_file)
    height, width, color = image.shape
    edge_h = int(round((height - 108) / 2.0))
    edge_w = int(round((width - 108) / 2.0))

    cropped = image[edge_h:(edge_h + 108), edge_w:(edge_w + 108)]
    small = imresize(cropped, (64, 64))

    filename = input_file.split('/')[-1]
    imsave("%s/%s" % (output_dir, filename), small)


def files2images(filenames):
    return [scale_image(imread(fn)) for fn in filenames]


def download_file_from_google_drive(file_id, dest):
    drive_url = "https://docs.google.com/uc?export=download"

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    session = requests.Session()
    response = session.get(drive_url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(drive_url, params=params, stream=True)

    def save_response_content(r, dest):
        # unfortunately content-length is not provided in header
        total_iters = 1409659  # in KB
        with open(dest, 'wb') as f:
            for chunk in tqdm(
                    r.iter_content(1024),
                    total=total_iters,
                    unit='KB',
                    unit_scale=True):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    save_response_content(response, dest)
