import os

def count_images_in_directory(directory_path):
    # Check if the path is a valid directory
    if not os.path.isdir(directory_path):
        raise ValueError("Invalid directory path")

    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Filter only files with image extensions (you may need to modify this based on your dataset)
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Count the number of image files
    num_images = len(image_files)

    return num_images

# Example usage:
dataset_path = "images"
image_count = count_images_in_directory(dataset_path)
print(f"Number of images in the dataset: {image_count}")

'''import os
import shutil
import random

def split_dataset(source_directory, destination_directory, target_num_images):
    # Check if the source path is a valid directory
    if not os.path.isdir(source_directory):
        raise ValueError("Invalid source directory path")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Get a list of all files in the source directory
    files = os.listdir(source_directory)

    # Filter only files with image extensions (you may need to modify this based on your dataset)
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Select the first target_num_images images
    selected_images = image_files[:target_num_images]

    # Copy the selected images to the destination directory
    for image in selected_images:
        source_path = os.path.join(source_directory, image)
        destination_path = os.path.join(destination_directory, image)
        shutil.copyfile(source_path, destination_path)

    print(f"Successfully created a subset of {target_num_images} images in the destination directory.")

# Example usage:
source_dataset_path = "photos"
destination_dataset_path = "images"
target_image_count = 10000

split_dataset(source_dataset_path, destination_dataset_path, target_image_count)

import csv'''

'''def count_csv_rows(csv_file_path):
    # Open the CSV file in read mode
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Use the len() function to count the number of rows
        row_count = len(list(csv_reader))

    return row_count

# Example usage:
csv_file_path = "styles.csv"
rows_count = count_csv_rows(csv_file_path)
print(f"Number of rows in the CSV file: {rows_count}")'''

'''import csv
import random

def split_csv(input_csv_path, output_csv_path, target_num_records):
    # Open the original CSV file in read mode
    with open(input_csv_path, 'r') as input_file:
        # Create a CSV reader
        csv_reader = csv.reader(input_file)

        # Read all rows from the original CSV file
        all_rows = list(csv_reader)

        # Shuffle the rows
        random.shuffle(all_rows)

        # Select the first target_num_records rows
        selected_rows = all_rows[:target_num_records]

    # Write the selected rows to a new CSV file
    with open(output_csv_path, 'w', newline='') as output_file:
        # Create a CSV writer
        csv_writer = csv.writer(output_file)

        # Write the selected rows to the new CSV file
        csv_writer.writerows(selected_rows)

    print(f"Successfully created a subset of {target_num_records} records in the output CSV file.")

# Example usage:
input_csv_path = "/path/to/your/original/file.csv"
output_csv_path = "/path/to/your/new/file.csv"
target_record_count = 10000

split_csv(input_csv_path, output_csv_path, target_record_count)
'''

'''import pandas as pd
import random
import os

def select_subset(images_folder, styles_csv_path, output_images_csv_path, output_styles_csv_path, target_num_images):
    # Get a list of all image filenames in the images folder
    image_files = os.listdir(images_folder)

    # Create a DataFrame with the image filenames
    images_df = pd.DataFrame({"Id": [filename.split('.')[0] for filename in image_files]})

    # Read the styles CSV file into a DataFrame
    styles_df = pd.read_csv(styles_csv_path)

    # Merge the two DataFrames based on the "Id" column
    merged_df = pd.merge(images_df, styles_df, on="id")

    # Shuffle the merged DataFrame
    shuffled_df = merged_df.sample(frac=1, random_state=42)

    # Select the first target_num_images rows
    selected_df = shuffled_df.head(target_num_images)

    # Split the selected DataFrame into images and styles DataFrames
    selected_images_df = selected_df[["Id"]]
    selected_styles_df = selected_df.drop(columns=["Id"])

    # Write the selected images DataFrame to a new CSV file
    selected_images_df.to_csv(output_images_csv_path, index=False)

    # Write the selected styles DataFrame to a new CSV file
    selected_styles_df.to_csv(output_styles_csv_path, index=False)

    print(f"Successfully created subsets of {target_num_images} images and their styles in the output CSV files.")

# Example usage:
images_folder = "photos"
styles_csv_path = "style.csv"
output_images_csv_path = "/path/to/your/images_subset.csv"
output_styles_csv_path = "/path/to/your/styles_subset.csv"
target_image_count = 10000

select_subset(images_folder, styles_csv_path, output_images_csv_path, output_styles_csv_path, target_image_count)
'''

'''
import pandas as pd
import os

# Load your CSV file
df = pd.read_csv(r"style.csv")

# Path to your images folder
images_folder = 'images'

# List all image files in the folder
image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

# Filter the DataFrame based on the image IDs in your images folder
filtered_df = df[df['id'].astype(str).str.replace('.jpg', '').isin(image_files)]

# Save the filtered DataFrame to a new CSV file
filtered_csv_path = 'styles.csv'
filtered_df.to_csv(filtered_csv_path, index=False)'''
