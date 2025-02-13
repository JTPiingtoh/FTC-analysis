import os
from tqdm import tqdm
import pandas as pd
from analysis import full_analysis
import tkinter as tk
from tkinter import filedialog
from tifffile import TiffFile
from roifile import ImagejRoi
from new_analysis import FTC_analysis 

def get_folder_directory():
    root = tk.Tk()
    root.withdraw() 

    folder_path = filedialog.askdirectory(title="Select a folder")

    if not folder_path:
        print("No folder selected or operation cancelled.")
        return None

    print("Selected folder:", folder_path)

    root.destroy()  
    return folder_path


if __name__ == "__main__":
    input_directory = get_folder_directory()

    if input_directory:
        
        # Create new folder to store Excel/csv and analysed images
        input_file_name = os.path.basename(input_directory)

        output_base_dir = os.path.dirname(input_directory)
        output_dir = os.path.join(output_base_dir, f"{input_file_name} OUTPUTS")
        
        try:
            os.makedirs(output_dir)

            # Create folder to store analysed images in
            anaylsed_images_dir = os.path.join(output_dir, f"{input_file_name} ANALYSED IMAGES")
            os.makedirs(anaylsed_images_dir)

        except OSError as e:
            print(f"An output folder for the selected folder may already exist. Check to see if {output_dir} already exists on your machine; if it does you may have already ran analyis.")
            exit()


    results_list = []

    '''
    07/02/2025 - Improved path strings, should resolve permission error alongside writing to csv. If this code still break, use os.chmod 664
    '''
    for file in os.listdir(input_directory):

        image_path = os.path.join(input_directory, file).replace('\\', '/')
        
        filename, extension = os.path.splitext(file)

        # Attempt to open tiff file TODO: List non tif files in dir
        if not extension.lower() in ('.tif', 'tiff'):
            print(f"{file} is not a tif file.")
            continue
            
        # Attempt to parse ROI
        with TiffFile(image_path) as tif:
            try:          
                image_array = tif.pages[0].asarray()
                roi_bytes = tif.imagej_metadata['ROI'] # KeyError
                roi = ImagejRoi.frombytes(roi_bytes)
                FTC_results_dict = FTC_analysis(image_array=image_array, roi=roi)



            except KeyError:
                print(f"{file} has no ROI.")
                continue 
    
    #     result = full_analysis(image_path=image_path)
    #     results_list.append(result)
    #     analysed_image_path = os.path.join(anaylsed_images_dir, f"{filename} ANALYSED{extension}").replace('\\', '/')
    #     result["img"].save(analysed_image_path)

    # # Create a DataFrame from the results list
    # results_df = pd.DataFrame(results_list)

    # csv_file_path = os.path.join(output_dir, f"{input_file_name} ANALYSED.csv").replace('\\', '/')
    # results_df.to_csv(csv_file_path, index=False)

    # os.startfile(output_base_dir)

        
