import pandas as pd
from PIL import Image
import io
import os
import ast  # For safely evaluating stringified dictionaries

# Load the CSV file
csv_file = "data.csv" 
df = pd.read_csv(csv_file)

# Print column names and the first row
print("Columns:", df.columns)
print("First row:", df.iloc[0])


# Load the CSV file
csv_file = "data.csv"  # Update with your file path
df = pd.read_csv(csv_file)

# Create a directory to save images
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Process each row
for idx, row in df.iterrows():
    try:
        # The 'image' column contains a dictionary as a string, e.g., "{'bytes': b'...'}"
        image_data = row["image"]
        
        # If image_data is a string, it’s likely a stringified dictionary
        if isinstance(image_data, str):
            # Safely evaluate the string to a dictionary
            image_dict = ast.literal_eval(image_data)
            if isinstance(image_dict, dict) and "bytes" in image_dict:
                image_bytes = image_dict["bytes"]
            else:
                print(f"Skipping row {idx}: No 'bytes' key in image dictionary")
                continue
        elif isinstance(image_data, dict):
            # If already a dict (unlikely, but possible depending on CSV parsing)
            image_bytes = image_data["bytes"]
        else:
            print(f"Skipping row {idx}: Image data is not in expected format")
            continue

        # Verify the bytes start with PNG header
        if not image_bytes.startswith(b'\x89PNG'):
            print(f"Skipping row {idx}: Image bytes do not start with PNG header")
            continue

        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Verify image integrity
        image.verify()
        # Reopen the image after verify (verify closes the file)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save as PNG
        image.save(f"{output_dir}/image_{idx}.png")
        print(f"Saved image {idx}")

    except Exception as e:
        print(f"Error processing image {idx}: {e}")

print(f"Images saved to {output_dir}")