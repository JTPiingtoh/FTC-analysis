from PIL import ImageDraw
import math
import os
from rotate_trim import us_rotater_and_trim
from typing import List, Tuple

# UPDATED: 13/02/24 12:37


def full_analysis(image_path):
    # record image name.png
    image_name = os.path.basename(image_path)
    # OPTIONAL: add image rotator and trim here
    rotated_image_and_trimmed = us_rotater_and_trim(image_path)

    # Open the image
    img = rotated_image_and_trimmed

    # record image name for spreadsheet
    file_name_no_extension = os.path.splitext(image_name)[0]

    # Convert the image to RGB mode (if not already in RGB)
    img = img.convert("RGB")

    # Get the image's pixel map
    pixels = img.load()

    # Define the RGB values for white and red
    white_color = (255, 255, 255)
    intercondyl_colour = red_color = (255, 0, 0)
    medial_color = green_color = (0, 200, 0)
    lateral_color = blue_color = (50, 50, 255)

    pink_color = (200, 0, 200)
    orange_color = (255, 127, 0)
    trim_color = teal_color = (100, 120, 140)

    grey_color = outside_color = (10, 12, 14)

    medial_zone_color = (0, 200, 255)

    lateral_zone_color = (255, 200, 255)
    intercondyl_zone_color = (255, 200, 0)

    # For converting between pixels and mm
    def pixels_to_mm(pixels):
        mm = pixels * 10 * 3.5 / 454
        return mm

    def mm_to_pixels(mm):
        pixels = ((mm / 10) * 454)/3.5
        return pixels

    width, height = img.size

    # Finding top and bottom line coords, with Lisee et al segments defined aswell
    ####################################################################################################################
    ####################################################################################################################
    # Find Top line coords

    trimmed_top_pixel_coords = []
    for x in range(width):
        for y in range(height - 1):
            pixel_value = img.getpixel((x, y))
            next_pixel_value = img.getpixel((x, y + 1))
            if pixel_value == white_color and next_pixel_value != white_color:

                if next_pixel_value != trim_color:
                    trimmed_top_pixel_coords.append((x, y))
                    # pixels[x, y] = green_color
                    break

    ####################################################################################################################
    ####################################################################################################################
    # FOR FINDING MIDDLE POINT
    ####################################################################################################################

    # Find coordinates with the maximum y value
    max_y_value = max(trimmed_top_pixel_coords, key=lambda coord: coord[1])[1]
    max_y_coordinates = [(x, y) for x, y in trimmed_top_pixel_coords if y == max_y_value]

    # Extract unique x coordinates using a set
    unique_x_coordinates = set(x for x, y in max_y_coordinates)

    # Convert the set to a list and find the middle x coordinate
    unique_x_list = list(unique_x_coordinates)

    med_x_coordinate_no = len(unique_x_list) // 2
    med_x_coordinate = unique_x_list[med_x_coordinate_no]

    # Now we have the central x coordinate, we need to find the 3 25% zones based upon this number. We also need to
    # find the end zones based on the trimming that has already been done
    ###################################################################################################################
    # Extract x coordinates from top line
    trimmed_top_line_x_values = [x for x, y in trimmed_top_pixel_coords]

    # Find the value of the highest x coordinate
    trimmed_max_x_value = max(trimmed_top_line_x_values)

    # Find the value of the lowest x coordinate
    trimmed_min_x_value = min(trimmed_top_line_x_values)

    # Set the color of pixels with maximum y coordinates to green
    # pixels[med_x_coordinate, max_y_value - 1] = green_color

    # find working area based on width of CSA
    trimmed_working_width = trimmed_max_x_value - trimmed_min_x_value

    # Now we take these values and create our zones. These will be visualised later in the script.
    lisee_medial_line_x = med_x_coordinate - int((0.5 * (1 / 3.0)) * trimmed_working_width)
    lisee_lateral_line_x = med_x_coordinate + int((0.5 * (1 / 3.0)) * trimmed_working_width)

    # renaming for ease of use
    leftmost_analysis_line_x = trimmed_min_x_value
    rightmost_analysis_line_x = trimmed_max_x_value
    ####################################################################################################################

    # Now we find the bottom line coords
    ####################################################################################################################
    ####################################################################################################################

    bone_cartilage_medial_coords = []
    bone_cartilage_lateral_coords = []
    bone_cartilage_intercondyl_cords = []
    bone_cartilage_all_coords = []

    for x in range(width - 1, -1, -1):
        for y in range(height - 1, -1, -1):
            pixel_value = img.getpixel((x, y))
            next_pixel_value = img.getpixel((x, y - 1))

            if leftmost_analysis_line_x < x < rightmost_analysis_line_x:

                if pixel_value == white_color and next_pixel_value != white_color:

                    bone_cartilage_all_coords.append((x, y))

                    if x - 1 < lisee_medial_line_x:
                        bone_cartilage_medial_coords.append((x, y))  # add pixel coord to lower zone line
                        # pixels[x, y] = green_color  # color pixel green
                        break
                    elif x > lisee_lateral_line_x:
                        bone_cartilage_lateral_coords.append((x, y))
                        # pixels[x, y] = green_color
                        break
                    elif lisee_medial_line_x - 1 < x < lisee_lateral_line_x + 1:
                        bone_cartilage_intercondyl_cords.append((x, y))
                        # pixels[x, y] = blue_color
                        break

    ####################################################################################################################
    # Now we have a list of coordinates for the bottom and top lines. From these "Lisee lines" we can now calculate the
    # CSA from these point, as well as the average EI of each section.

    # Calculating average RGBs
    ####################################################################################################################
    ####################################################################################################################
    average_rgb_medial = [0, 0, 0]
    medial_pixel_count = 0

    average_rgb_lateral = [0, 0, 0]
    lateral_pixel_count = 0

    average_rgb_intercondyl = [0, 0, 0]
    intercondyl_pixel_count = 0

    # for calculating average rgb of each zone. This for loop will also count the number of pixels in each section which
    # can then be converted to average thickness
    for x in range(width):
        for y in range(height):
            pixel_value = img.getpixel((x, y))
            if pixel_value != white_color:
                if leftmost_analysis_line_x < x < lisee_medial_line_x:
                    medial_pixel_count += 1
                    rm2, gm2, bm2 = pixel_value
                    # pixels[x, y] = pink_color
                    rm1, gm1, bm1 = average_rgb_medial
                    average_rgb_medial = (
                        (rm1 * (medial_pixel_count - 1) + rm2) / medial_pixel_count,
                        (gm1 * (medial_pixel_count - 1) + gm2) / medial_pixel_count,
                        (bm1 * (medial_pixel_count - 1) + bm2) / medial_pixel_count
                    )

                if lisee_lateral_line_x < x < rightmost_analysis_line_x:
                    lateral_pixel_count += 1
                    rl2, gl2, bl2 = pixel_value
                    # pixels[x, y] = pink_color
                    rl1, gl1, bl1 = average_rgb_lateral
                    average_rgb_lateral = (
                        (rl1 * (lateral_pixel_count - 1) + rl2) / lateral_pixel_count,
                        (gl1 * (lateral_pixel_count - 1) + gl2) / lateral_pixel_count,
                        (bl1 * (lateral_pixel_count - 1) + bl2) / lateral_pixel_count
                    )

                if lisee_medial_line_x < x < lisee_lateral_line_x:
                    intercondyl_pixel_count += 1
                    ri2, gi2, bi2 = pixel_value
                    # pixels[x, y] = pink_color
                    ri1, gi1, bi1 = average_rgb_intercondyl
                    average_rgb_intercondyl = (
                        (ri1 * (intercondyl_pixel_count - 1) + ri2) / intercondyl_pixel_count,
                        (gi1 * (intercondyl_pixel_count - 1) + gi2) / intercondyl_pixel_count,
                        (bi1 * (intercondyl_pixel_count - 1) + bi2) / intercondyl_pixel_count
                    )
    total_pixels = medial_pixel_count + lateral_pixel_count + intercondyl_pixel_count

    # add different rgb values to seperate lists to make sure that only black and white image has been assessed.
    # Check this in the excel doc
    average_red_value_medial = average_rgb_medial[0]
    average_green_value_medial = average_rgb_medial[1]
    average_blue_value_medial = average_rgb_medial[2]

    average_red_value_lateral = average_rgb_lateral[0]
    average_green_value_lateral = average_rgb_lateral[1]
    average_blue_value_lateral = average_rgb_lateral[2]

    average_red_value_intercondyl = average_rgb_intercondyl[0]
    average_green_value_intercondyl = average_rgb_intercondyl[1]
    average_blue_value_intercondyl = average_rgb_intercondyl[2]

    def rgb_to_echo_intensity(rgb):
        # Luminosity method: Y = 0.299*R + 0.587*G + 0.114*B
        # grayscale_value = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])

        # Average method: Y = (R + G + B) / 3
        grayscale_value = sum(rgb) // 3
        # echo_intensity = 100 * (grayscale_value/255)
        echo_intensity = grayscale_value
        return echo_intensity

    # converting rgb values to grayscale
    lisee_average_medial_EI = rgb_to_echo_intensity(average_rgb_medial)
    lisee_average_lateral_EI = rgb_to_echo_intensity(average_rgb_lateral)
    lisee_average_intercondyl_EI = rgb_to_echo_intensity(average_rgb_intercondyl)

    # functions for calculating the length of cartilage sections
    def calculate_distance(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def total_distance(coordinates):
        total_distance_value = 0
        for i in range(len(coordinates) - 1):
            distance = calculate_distance(coordinates[i], coordinates[i + 1])
            total_distance_value += distance
        return total_distance_value

    # Calculate the total distances of cartilage zones
    medial_cartilage_length = total_distance(bone_cartilage_medial_coords)

    lateral_cartilage_length = total_distance(bone_cartilage_lateral_coords)

    intercondyl_cartilage_length = total_distance(bone_cartilage_intercondyl_cords)

    # calculate average thicknesses
    medial_average_px_thickness = medial_pixel_count / medial_cartilage_length
    lisee_medial_average_mm_thickness = pixels_to_mm(medial_average_px_thickness)

    lateral_average_px_thickness = lateral_pixel_count / lateral_cartilage_length
    lisee_lateral_average_mm_thickness = pixels_to_mm(lateral_average_px_thickness)

    intercondyl_average_px_thickness = intercondyl_pixel_count / intercondyl_cartilage_length
    lisee_intercondyl_average_mm_thickness = pixels_to_mm(intercondyl_average_px_thickness)

    ####################################################################################################################
    ####################################################################################################################
    # NEW SECTION
    ####################################################################################################################
    ####################################################################################################################
    # Now we need to calculate measurement of thickness and CSA from Horis's landmarks of 1 cm each side of the notch.

    # for finding medial lower pointx
    def get_coordinate_by_x(x_value, coordinates_list):
        matching_coordinates = [coord for coord in coordinates_list if coord[0] == x_value]

        if matching_coordinates:
            return matching_coordinates[0]
        else:
            return None  # Return None if no matching coordinate is found

    # For findning 2 closest points and the distance between them
    def find_closest_coordinate_and_distance(target_coord, coordinates_list):
        if not coordinates_list:
            return None, None  # Return None if the list is empty


        # Calculate distances to the target coordinate
        distances = [math.sqrt((x - target_coord[0]) ** 2 + (y - target_coord[1]) ** 2) for x, y in coordinates_list]

        # Find the index of the coordinate with the minimum distance
        min_distance_index = distances.index(min(distances))

        # Get the closest coordinate, its distance, and x coordinates
        closest_coord = coordinates_list[min_distance_index]
        closest_distance = distances[min_distance_index]

        # Return the closest coordinate, its distance, and x coordinates for both
        return closest_coord, closest_distance

    ####################################################################################################################
    # Get the image's drawing context
    draw = ImageDraw.Draw(img)

    ####################################################################################################################
    hori_medial_line_x = med_x_coordinate - int((mm_to_pixels(10)))
    hori_lateral_line_x = med_x_coordinate + int((mm_to_pixels(10)))

    leftmost_csa_coord = get_coordinate_by_x(int(hori_medial_line_x), bone_cartilage_all_coords)
    rightmost_csa_coord = get_coordinate_by_x(int(hori_lateral_line_x), bone_cartilage_all_coords)
    centre_csa_coord = get_coordinate_by_x(int(med_x_coordinate), bone_cartilage_all_coords)

    left_csa_clo_dis_tup = find_closest_coordinate_and_distance(leftmost_csa_coord, trimmed_top_pixel_coords)
    closest_top_coord_to_lftmst_csa_crd = left_csa_clo_dis_tup[0]

    right_csa_clo_dis_tup = find_closest_coordinate_and_distance(rightmost_csa_coord, trimmed_top_pixel_coords)
    closest_top_coord_to_rhtmst_csa_crd = right_csa_clo_dis_tup[0]

    centre_clo_dis_tup = find_closest_coordinate_and_distance(centre_csa_coord, trimmed_top_pixel_coords)
    closest_top_coord_to_cntr_crd = centre_clo_dis_tup[0]

    hori_medial_thickness_px = left_csa_clo_dis_tup[1]
    hori_medial_thickness_mm = pixels_to_mm(hori_medial_thickness_px)

    hori_lateral_thickness_px = right_csa_clo_dis_tup[1]
    hori_lateral_thickness_mm = pixels_to_mm(hori_lateral_thickness_px)

    hori_intercondyl_thickness_px = centre_clo_dis_tup[1]
    hori_central_thickness_mm = pixels_to_mm(hori_intercondyl_thickness_px)

    draw.line([leftmost_csa_coord, closest_top_coord_to_lftmst_csa_crd], fill="pink", width=2)
    draw.line([rightmost_csa_coord, closest_top_coord_to_rhtmst_csa_crd], fill="pink", width=2)
    draw.line([centre_csa_coord, closest_top_coord_to_cntr_crd], fill="pink", width=2)

    leftmost_max_x_topline = closest_top_coord_to_lftmst_csa_crd[0]
    rightmost_max_x_topline = closest_top_coord_to_rhtmst_csa_crd[0]

    def coord_filterby_x(x1, x2, coordinates):
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        filtered_coordinates = [coord for coord in coordinates if min_x <= coord[0] <= max_x]
        return filtered_coordinates

    def coord_add_and_organise_by_y(coordinate_list, new_coordinate):
        coordinate_list.append(new_coordinate)
        return sorted(coordinate_list, key=lambda coord: coord[1])

    def y_axis_integration(coordinates: List[Tuple[float, float]], x_offset: float) -> float:
        integral = 0.0
        for i in range(1, len(coordinates)):
            dy = coordinates[i][1] - coordinates[i - 1][1]
            integral += abs(((
                                     (coordinates[i][0] + coordinates[i - 1][0])
                                     / 2.0)
                             * dy)
                            - dy * x_offset)
        return integral

    leftmost_topline_coords = coord_filterby_x(hori_medial_line_x, leftmost_max_x_topline,
                                               trimmed_top_pixel_coords)
    leftmost_coords_for_int = coord_add_and_organise_by_y(leftmost_topline_coords, leftmost_csa_coord)

    for x, y in leftmost_coords_for_int:
        pixels[x, y] = red_color

    leftmost_px_area_trim = int(y_axis_integration(leftmost_coords_for_int, hori_medial_line_x))

    rightmost_topline_coords = coord_filterby_x(hori_lateral_line_x, rightmost_max_x_topline,
                                                trimmed_top_pixel_coords)
    rightmost_coords_for_int = coord_add_and_organise_by_y(rightmost_topline_coords, rightmost_csa_coord)

    for x, y in rightmost_coords_for_int:
        pixels[x, y] = green_color

    rightmost_px_area_trim = int(y_axis_integration(rightmost_coords_for_int, hori_lateral_line_x))

    leftmost_bottom_csa_x = leftmost_csa_coord[0]
    rightmost_bottom_csa_x = rightmost_csa_coord[0]

    #print(f"leftmost trim: {leftmost_px_area_trim}")
    #print(f"rightmost trim: {rightmost_px_area_trim}")

    raw_10mm_csa_px_count = 0
    for x in range(width):
        for y in range(height):
            pixel_value = img.getpixel((x, y))
            if pixel_value != white_color:
                if leftmost_bottom_csa_x <= x <= rightmost_bottom_csa_x:
                    raw_10mm_csa_px_count += 1
                    # pixels[x, y] = pink_color

    #print(f"raw csa: {raw_10mm_csa_px_count}")

    hori_csa_px_count = raw_10mm_csa_px_count - leftmost_px_area_trim - rightmost_px_area_trim
    hori_csa_mm = hori_csa_px_count * (pixels_to_mm(1) ** 2)

    ###################################################################################################################
    ###################################################################################################################



    # Medial condyl line
    for x in range(width):
        for y in range(height):
            pixels[lisee_medial_line_x, y] = green_color

    # Lateral condyl line

    for x in range(width):
        for y in range(height):
            pixels[lisee_lateral_line_x, y] = blue_color

    # Medial condyl line
    for x in range(width):
        for y in range(height):
            pixels[hori_medial_line_x, y] = orange_color

    # Lateral condyl line

    for x in range(width):
        for y in range(height):
            pixels[hori_lateral_line_x, y] = red_color

    return {
        "Image_name": file_name_no_extension,  # Include the image in the result

        "Hori_medial_thickness_mm": hori_medial_thickness_mm,
        "Hori_lateral_thickness_mm": hori_lateral_thickness_mm,
        "Hori_central_thickness_mm": hori_central_thickness_mm,
        "Hori_csa_mm": hori_csa_mm,

        "Lisee_Medial_average_mm_thickness": lisee_medial_average_mm_thickness,
        "Lisee_Lateral_average_mm_thickness": lisee_lateral_average_mm_thickness,
        "Lisee_Intercondyl_average_mm_thickness": lisee_intercondyl_average_mm_thickness,

        "Lisee_Medial_average_EI": lisee_average_medial_EI,
        "Lisee_Lateral_average_EI": lisee_average_lateral_EI,
        "Lisee_Intercondyl_average__EI": lisee_average_intercondyl_EI,

        "<": "#",
        "Total_px_count": total_pixels,
        ">": "#",

        "Lisee_red_value_medial": average_red_value_medial,
        "Lisee_green_value_medial": average_green_value_medial,
        "Lisee_blue_value_medial": average_blue_value_medial,

        "Lisee_red_value_lateral": average_red_value_lateral,
        "Lisee_green_value_lateral": average_green_value_lateral,
        "Lisee_blue_value_lateral": average_blue_value_lateral,

        "Lisee_red_value_intercondyl": average_red_value_intercondyl,
        "Lisee_green_value_intercondyl": average_green_value_intercondyl,
        "Lisee_blue_value_intercondyl": average_blue_value_intercondyl,

        "img": img,
    }