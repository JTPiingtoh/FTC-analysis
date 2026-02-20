import os
import platform
import tkinter as tk
from tkinter import filedialog
import logging

import matplotlib.pyplot as plt
import pandas as pd
from roifile import ImagejRoi
from tifffile import TiffFile, TiffFileError
from tqdm import tqdm
from datetime import datetime

from ftc_helpers.ftc_analysis import FTC_analysis

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
        output_dir = os.path.join(output_base_dir, f"{input_file_name} OUTPUTS - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}").replace('\\', '/')
        
        
        try:
            os.makedirs(output_dir)
            analysed_images_dir = os.path.join(output_dir, f"{input_file_name} ANALYSED IMAGES")        
            os.makedirs(analysed_images_dir, exist_ok=True)
            logging_file_name = os.path.join(output_dir, "analysis_log.log")
            logging.basicConfig(
                filename=logging_file_name,
                level=logging.INFO
            )

            logger= logging.getLogger(__name__)
        except OSError as e:
            print(f"Fatal error setting up output files: {e}")
            print("Terminating program.")
            os.removedirs(output_dir)

        results_list = []
        # 07/02/2025 - Improved path strings, should resolve permission error alongside writing to csv. 
        # If this code still breaks, use os.chmod 664
        
        files_to_analyse = os.listdir(input_directory)
        n_files = len(files_to_analyse)
        n_files_analysed = 0
        print("Starting anaylsis")
        for file in tqdm(files_to_analyse):

            image_path = os.path.join(input_directory, file).replace('\\', '/')
            filename, extension = os.path.splitext(file)

            # Attempt to open tiff file TODO: List non tif files in dir
            if extension.lower() not in ('.tif', '.tiff'):
                logger.info(f"{file} is not a .tiff file. Analysis was skipped")
                continue
                
            # Attempt to parse ROI
            with TiffFile(image_path) as tif:
                try:          
                    image_array = tif.pages[0].asarray()
                    roi_bytes = tif.imagej_metadata['ROI'] 
                    roi = ImagejRoi.frombytes(roi_bytes)
                    FTC_results_dict, FTC_img = FTC_analysis(image_array=image_array, roi=roi, filename=filename)
                    results_list.append(FTC_results_dict)
                    analysed_image_path = os.path.join(analysed_images_dir, f"{filename} ANALYSED.png").replace('\\', '/')
                    FTC_img.savefig(analysed_image_path)
                    plt.close(FTC_img)
                    n_files_analysed += 1

                except KeyError as e:
                    logging.info(f"{filename}: {e}")
                    continue 
                except ValueError as e:
                    logging.info(f"{filename}: {e}")
                    continue 
                except RuntimeError as e:
                    logging.info(f"{filename}: {e}")
                    continue 
                except TiffFileError as e:
                    logging.info(f"{filename} has invalid stucture: {e}")
                    continue

        results_df = pd.DataFrame(results_list)
        csv_file_path = os.path.join(output_dir, f"{input_file_name} ANALYSED.csv").replace('\\', '/')
        results_df.to_csv(csv_file_path, index=False)
        logger.info(f"Files analysed: {n_files_analysed}/{n_files}")
        # open the output dir (WINDOWS ONLY)
        if platform.system() == 'Windows':
            os.startfile(output_dir)

    
    else:
        print("Program finished without analysis")


        
