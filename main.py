import os
from tqdm import tqdm
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tifffile import TiffFile, TiffFileError
from roifile import ImagejRoi
from ftc_analysis import FTC_analysis 
import matplotlib.pyplot as plt

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
        output_dir = os.path.join(output_base_dir, f"{input_file_name} OUTPUTS").replace('\\', '/')
        
        try:
            # Create folder to store analysed images in
            anaylsed_images_dir = os.path.join(output_dir, f"{input_file_name} ANALYSED IMAGES")        
            os.makedirs(output_dir)
            os.makedirs(anaylsed_images_dir)

        # TODO: change this to creating a metadata file in the image dir
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
                print(filename)
                image_array = tif.pages[0].asarray()
                roi_bytes = tif.imagej_metadata['ROI'] 
                roi = ImagejRoi.frombytes(roi_bytes)
                FTC_results_dict = FTC_analysis(image_array=image_array, roi=roi)
                FTC_results_dict["image_name"] = filename
                results_list.append(FTC_results_dict)

                analysed_image_path = os.path.join(anaylsed_images_dir, f"{filename} ANALYSED{extension}").replace('\\', '/')
          
                FTC_results_dict["img"].savefig(analysed_image_path)
                plt.close(FTC_results_dict["img"])

            except KeyError as e:
                print(f"{filename}: {e}")
                continue 
            except ValueError as e:
                print(f"{filename}: {e}")
                continue 
            except RuntimeError as e:
                print(f"{filename}: {e}")
                continue 
            except TiffFileError as e:
                print(f"{filename}: {e}")
                continue

        
    results_df = pd.DataFrame(results_list)
    csv_file_path = os.path.join(output_dir, f"{input_file_name} ANALYSED.csv").replace('\\', '/')
    results_df.to_csv(csv_file_path, index=False)

    os.startfile(output_base_dir)

        
