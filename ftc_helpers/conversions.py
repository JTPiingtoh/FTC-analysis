def mm_to_pixels(
    mm: float|int,
    image_height_px: float|int = 454,
    image_height_cm: float|int = 3.5,
    ) -> float:   
    return ((mm / 10) * image_height_px) / image_height_cm

def pixels_to_mm(
    pixels: float|int,
    image_height_px: float|int = 454,
    image_height_cm: float|int = 3.5
    ) -> float:
    return (pixels * 
                    10 * # convert to mm 
                            image_height_cm) / image_height_px 