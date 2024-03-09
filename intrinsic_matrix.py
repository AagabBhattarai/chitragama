import numpy as np
from PIL import Image

def compute_intrinsic_matrix(image_path, database_path):
    img = Image.open(image_path)
    
    def get_info_from_exif(img):
        exif_data = img._getexif()
        focal_length_tag = 37386
        focal_length = exif_data.get(focal_length_tag)
        model_tag = 272
        model_name = exif_data[model_tag].title()
        width, height = img.size
        return model_name, focal_length, width, height
    
    def read_from_database(file_path, model_name):
        with open(file_path, "r") as file:
            file_lines = file.readlines()
        for line in file_lines:
            parts = line.split(",")
            name = " ".join(parts[:2])
            model_name = model_name.replace("-", " ")
            if model_name == name:
                sensor_width = parts[3].strip()
                sensor_height = parts[4].strip()
                return float(sensor_width), float(sensor_height)
        raise ValueError("Sensor data not found for model: {}".format(model_name))
    
    try:
        model_name, focal_length_mm, image_width, image_height = get_info_from_exif(img)
        sensor_width, sensor_height = read_from_database(database_path, model_name)
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
