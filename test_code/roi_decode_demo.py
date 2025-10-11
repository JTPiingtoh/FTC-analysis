import numpy as np
from tifffile import TiffFile

with TiffFile("503 ana.tif") as tif:
    metadata = tif.imagej_metadata
    roi = metadata['ROI']
    bytes = int.from_bytes(roi[64:])
    print(bytes)