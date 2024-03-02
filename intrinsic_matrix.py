import numpy as np
from PIL import Image

def compute_intrinsic_matrix(image_path, database_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            raise ValueError("No Exif metadata found in the image.")

        def get_info_from_exif(img):
            focal_length_tag = 37386
            focal_length = exif_data.get(focal_length_tag)
            model_tag = 272
            model_name = exif_data.get(model_tag)
            if model_name is None:
                raise ValueError("Model name not found in the Exif metadata.")
            model_name = model_name.title()  # Convert model name to title case
            width, height = img.size
            return model_name, focal_length, width, height

        def read_from_database(file_path, model_name):
            with open(file_path, "r") as file:
                file_lines = file.readlines()
            for line in file_lines:
                if model_name in line:
                    parts = line.split(";")
                    sensor_width = parts[1].strip()
                    return float(sensor_width)
            raise ValueError("Sensor width not found for model: {}".format(model_name))
        
        # Call the helper functions and compute the intrinsic matrix
        model_name, focal_length_mm, image_width, image_height = get_info_from_exif(img)
        sensor_width = read_from_database(database_path, model_name)
        sensor_height = (sensor_width * image_height) / image_width

        # Focal lengths in pixels
        focal_length_x = (focal_length_mm * image_width) / sensor_width
        focal_length_y = (focal_length_mm * image_height) / sensor_height

        c_x = image_width / 2
        c_y = image_height / 2
        s = 0
        # Intrinsic matrix
        intrinsic_matrix = np.array([[focal_length_x, s, c_x],
                                     [0, focal_length_y, c_y],
                                     [0, 0, 1]])
        return intrinsic_matrix

    except Exception as e:
        print("Error:", e)
        return None
    
# intrinsic_matrix = compute_intrinsic_matrix("image1.jpg", "sensor_width_camera_database.txt")
