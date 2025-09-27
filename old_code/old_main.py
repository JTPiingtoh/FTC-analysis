import os
from tqdm import tqdm
import pandas as pd
from old_code.analysis import full_analysis
import tkinter as tk
from tkinter import filedialog


# UPDATED: 07/02/2024 22:24

def get_folder_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a folder dialog to choose a folder
    folder_path = filedialog.askdirectory(title="Select a folder")

    if not folder_path:
        print("No folder selected or operation cancelled.")
        return None

    print("Selected folder:", folder_path)

    root.destroy()  # Explicitly destroy the Tkinter window
    return folder_path


# For opening image directory once analysis is completed
def open_directory_in_explorer(directory_path):
    os.startfile(directory_path)


if __name__ == "__main__":
    input_directory = get_folder_directory()

    if input_directory:
        # Continue with the rest of your script
        output_base_directory = os.path.dirname(input_directory)
        # os.makedirs(output_base_directory, exist_ok=True)

        # New folder name
        input_file_name = os.path.basename(input_directory)

        # Create a new folder for modified images
        output_images_directory = os.path.join(output_base_directory, f"ANALYSED {input_file_name}")
        os.makedirs(output_images_directory, exist_ok=True)

        # Results list to collect data for each image
        results_list = []


# Iterate over PNG images in the input directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith(".png") or filename.endswith(".bmp"):
        image_path = os.path.join(input_directory, filename)

        # OPTIONAL: add image rotator here

        # Apply the analysis function
        result = full_analysis(image_path)

        # Append results to the list
        results_list.append(result)

        # Save the modified image to the new folder
        modified_image_path = os.path.join(output_images_directory, f"ANALYSED {filename}")
        result["img"].save(modified_image_path)

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# Write the DataFrame to an Excel file
excel_file_path = os.path.join(output_base_directory, f"{input_file_name} ANALYSED.xlsx")
results_df.to_excel(excel_file_path, index=False)

print(f"Analysis results saved to {excel_file_path}")
print(f"Modified images saved to {output_images_directory}")

open_directory_in_explorer(output_base_directory)