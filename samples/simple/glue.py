import requests
import os
from PIL import Image
from io import BytesIO
import math
from algorithm import algorithm


APP_URL = os.environ("EMPAIA_APP_API")
JOB_ID = os.environ("EMPAIA_JOB_ID")
TOKEN = os.environ("EMPAIA_TOKEN")
HEADER = {"Authorization": f"Bearer {TOKEN}"}


class WSIMeta:
    """
    Convenience wrapper for accessing slide metadata.
    """

    def __init__(self, response_json):
        self.width = response_json['extent']['x']
        self.height = response_json['extent']['y']
        self.tile_width = response_json['tile_extent']['x']
        self.tile_height = response_json['tile_extent']['y']
        self.wsi_id = response_json['id']
        self.levels = []
        for o in response_json['levels']:
            e = {}
            e['width'] = o['extent']['x']
            e['height'] = o['extent']['y']
            e['factor'] = o['downsample_factor']
            self.levels.append(e)

    def no_of_levels(self) -> int:
        return len(self.levels)

    def no_of_x_tiles(self, on_level: int) -> int:
        w = self.levels[on_level]['width']
        return math.ceil(w / self.tile_width)

    def no_of_y_tiles(self, on_level: int) -> int:
        h = self.levels[on_level]['height']
        return math.ceil(h / self.tile_height)

    def no_of_tiles(self, on_level: int) -> int:
        return self.no_of_x_tiles(on_level) * self.no_of_y_tiles(on_level)


class EmpaiaRect:

    def __init__(self, json_rect):
        self.id = json_rect['id']
        self.start_x = json_rect['upper_left'][0]
        self.start_y = json_rect['upper_left'][1]
        self.width = json_rect['width']
        self.height = json_rect['height']


def get_slide_meta(wsi_param_name: str) -> WSIMeta:
    input_url = f"{APP_URL}/v3/{JOB_ID}/inputs/{wsi_param_name}"
    r = requests.get(input_url, headers=HEADER)
    r.raise_for_status()
    return WSIMeta(r.json())


def get_roi_param(roi_param_name: str) -> WSIMeta:
    input_url = f"{APP_URL}/v3/{JOB_ID}/inputs/{roi_param_name}"
    r = requests.get(input_url, headers=HEADER)
    r.raise_for_status()
    return EmpaiaRect(r.json())


def get_slide_tile(wsi_id: str, level: int, slide_x: int, slide_y: int) -> Image:
    tile_url = f"{APP_URL}/v3/{JOB_ID}/tiles/{wsi_id}/level/{level}/position/{slide_x}/{slide_y}"
    r = requests.get(tile_url, headers=HEADER)
    r.raise_for_status()
    i = Image.open(BytesIO(r.content))
    return i


def get_slide_region(wsi_id: str, level: int, start_x: int, start_y: int, width: int, height: int) -> Image:
    region_url = f"{APP_URL}/v3/{JOB_ID}/regions/{wsi_id}/level/{level}/start/{start_x}/{start_y}/size/{width}/{height}"
    r = requests.get(region_url, headers=HEADER)
    r.raise_for_status()
    i = Image.open(BytesIO(r.content))
    return i


def send_result(output_param_name: str, reference_id: str, result_value: float):
    payload = {
        "name": "User readable description, e.g. quantification result",  # change to your preferences
        "descriptuin": "Optional description",
        "creator_type": "job",
        "creator_id": JOB_ID,
        "type": "float",
        "value": result_value,
        "reference_type": "rectangle",
        "reference_id": reference_id
    }
    url = f"{APP_URL}/v3/{JOB_ID}/outputs/{output_param_name}"
    r = requests.post(url, json=payload, headers=HEADER)
    r.raise_for_status()


def finish_job():
    url = f"{APP_URL}/v3/{JOB_ID}/finalize"
    r = requests.put(url, headers=HEADER)
    r.raise_for_status()


if __name__ == "__main__":
    slide_param_name = "my_wsi"
    roi_param_name = "my_roi"
    result_param_name = "my_quantification_result"
    slide_meta = get_slide_meta(slide_param_name)
    roi_meta = get_roi_param(roi_param_name)
    lvl = len(slide_meta.levels) - 1
    downloaded_slides = []
    for y in range(0, slide_meta.no_of_y_tiles(lvl)):
        for x in range(0, slide_meta.no_of_x_tiles(lvl)):
            downloaded_slides.append(get_slide_tile(slide_meta.wsi_id, lvl, x, y))
    result = algorithm(downloaded_slides, (roi_meta.start_x, roi_meta.start_y, roi_meta.width, roi_meta.height))
    send_result(result_param_name, roi_meta.id, result)
    finish_job()


