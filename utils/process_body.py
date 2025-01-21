from pathlib import Path
import sys
FILE = Path(__file__).resolve()
sys.path.append(str(FILE.parents[1]))
from ultralytics.data.augment import LetterBox

def pre_transform(im, imgsz, model):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = len({x.shape for x in im}) == 1
    letterbox = LetterBox(imgsz, auto=same_shapes and model.pt, stride=model.stride)
    return [letterbox(image=x) for x in im]

def main():
    pre_transform()