import os
from tqdm import tqdm
import pandas as pd
from old_code.analysis import full_analysis
import tkinter as tk
from tkinter import filedialog


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
            print(f'''An output folder for the selected folder may already exist. Check to see if "{output_dir}" already exists on your machine; if it does you may have already ran analyis. Delete {output_dir} and try again if not.''')
            exit()


    results_list = []

    '''
    07/02/2025 - Improved path strings, should resolve permission error alongside writing to csv. If this code still break, use os.chmod 664
    '''
    for file in tqdm(os.listdir(input_directory)):

        image_path = os.path.join(input_directory, file).replace('\\', '/')
        print(file)
        print(image_path)
        
        filename, extension = os.path.splitext(file)        
        result = full_analysis(image_path=image_path)
        results_list.append(result)
        analysed_image_path = os.path.join(anaylsed_images_dir, f"{filename} ANALYSED{extension}").replace('\\', '/')
        result["img"].save(analysed_image_path)

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)

    csv_file_path = os.path.join(output_dir, f"{input_file_name} ANALYSED.csv").replace('\\', '/')
    results_df.to_csv(csv_file_path, index=False)

    os.startfile(output_base_dir)

        
