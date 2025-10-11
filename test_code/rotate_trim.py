from PIL import Image, ImageDraw
import math
import os


# UPDATED: 07/02/2024 22:24
def us_rotater_and_trim(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to RGB mode (if not already in RGB)
    img = img.convert("RGB")

    # Define the RGB values for white and red
    red_color = (255, 0, 0)
    white_color = (255, 255, 255)
    trim_color = teal_color = (100, 120, 140)
    width, height = img.size
    grey_color = outside_color = (10, 12, 14)
    
    # Iterate through each top line pixel to get top line coordinates
    top_pixel_coords = []
    for x in range(width):
        for y in range(height - 1):
            pixel_value = img.getpixel((x, y + 1))
            if pixel_value != white_color:
                top_pixel_coords.append((x, y))
                #pixels[x, y] = red_color
                break

    # trim top line coords to remove ends
    exclude_count = int(len(top_pixel_coords) * 0.10)

    # Remove the first and last 10% using list slicing
    trimmed_top_coords = top_pixel_coords[exclude_count:-exclude_count]

    # If the rightmost and leftmost coordinates are the same, return the original image
    if len(trimmed_top_coords) < 2:
        return img

    rightmost_coord = trimmed_top_coords[0]
    leftmost_coord = trimmed_top_coords[-1]

    def angle_between_points(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2

        angle = math.atan2(y2 - y1, x2 - x1)
        angle = math.degrees(angle)
        return angle

    # Finding angle to adjust by

    adjustment_angle = angle_between_points(rightmost_coord, leftmost_coord)
    adjustment_angle = round(adjustment_angle)

    # changing angle based on top line coords
    img = img.rotate(adjustment_angle, expand=False, fillcolor=white_color)

    # Get the image's pixel map
    pixels = img.load()

    # Iterate through each top line pixel to get top line coordinates
    new_top_pixel_x_coords = []
    for x in range(width):
        for y in range(height - 1):
            pixel_value = img.getpixel((x, y + 1))
            if pixel_value != white_color:
                new_top_pixel_x_coords.append(x)
                #pixels[x, y] = red_color
                break

    # trim top line coords to remove ends
    new_exclude_count = int(len(new_top_pixel_x_coords) * 0.125)

    # Remove the first and last 12.5% using list slicing
    new_trimmed_top_x_coords = new_top_pixel_x_coords[new_exclude_count:-new_exclude_count]

    # If the rightmost and leftmost coordinates are the same, return the original image
    if len(new_trimmed_top_x_coords) < 2:
        return img

    new_leftmost_coord_x = new_trimmed_top_x_coords[0]
    new_rightmost_coord_x = new_trimmed_top_x_coords[-1]

    for x in range(width):
        for y in range(height):
            pixel_value = img.getpixel((x, y))
            if pixel_value != white_color:
                if x < new_leftmost_coord_x:
                    pixels[x, y] = trim_color

                elif x > new_rightmost_coord_x:
                    pixels[x, y] = trim_color

    return img
