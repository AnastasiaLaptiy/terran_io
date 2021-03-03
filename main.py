import click
import os
from datetime import datetime

from pathlib import Path
from scipy.spatial.distance import cosine

from terran.face import extract_features, face_detection
from terran.io import open_image, resolve_images
from terran.vis import display_image, vis_faces

threshold = 0.5
display = True
imagePath = open_image('path_to_image')
faces_in_reference = face_detection(imagePath)

ref_feature = extract_features(imagePath, faces_in_reference[0])
path = os.path.abspath(os.getcwd()) + "\src\input_photos"
click.echo(path)
paths = resolve_images(path, batch_size = 1)

for batch_paths in paths:
        batch_images = list(map(open_image, batch_paths))
        
        faces_per_image = face_detection(batch_images)      
        # for path, face in zip(batch_paths, faces_per_image):
        #         face_path = os.path.join("../terran_io/src/detected_faces/", datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + ".json")    
        #         file_faces = open(face_path, "a")
        #         file_faces.write(str(path) + "\n")
        #         file_faces.write(str(face))  
                    
        features_per_image = extract_features(batch_images, faces_per_image)         
        # for path, feature in zip(batch_paths, features_per_image):
        #         feature_path = os.path.join("../terran_io/src/features/", datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + ".json")    
        #         file_features = open(feature_path, "a")
        #         file_features.write(str(path) + "\n")
        #         file_features.write(str(feature))

        for path, image, faces, features in zip(
            batch_paths, batch_images, faces_per_image, features_per_image
        ):
            for face, feature in zip(faces, features):            
                confidence = cosine(ref_feature, feature)
                if confidence < threshold:
                    click.echo(f'{path}, confidence = {confidence:.2f}')
                    if display:
                        display_image(vis_faces(image, face))
                        