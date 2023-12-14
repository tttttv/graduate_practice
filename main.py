import os

from PIL import Image, ImageDraw

from networks.models import *

models = [
    Facenet(), VGGFace(), DeepID(), OpenFace()
]
test_image_folder = "data/test_images/"

for model in models:
    print('RUNNING MODEL', model.model_name)
    dr = os.listdir(test_image_folder)
    for image in dr:
        print('IMAGE', image)
        image_path = test_image_folder + image

        face = model.extract_faces(img_path = image_path)[0]
        print('detected face', face)
        facial_area = face['facial_area']
        face_coords = [facial_area['x'], facial_area['y'], facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']]

        result = model.find(img_path = image_path, db_path = "data/train_images")

        print('RES2', result)

        try:
            closest = result[0].values[0]
            name = closest[0].split('/')[2]
            distance = closest[-1]
        except IndexError: #Не распознано
            name = 'undefined'
            distance = 0


        # Draw faces
        img = Image.open(image_path)
        frame_draw = img.copy()
        draw = ImageDraw.Draw(frame_draw)
        draw.rectangle(face_coords, outline=(255, 0, 0), width=6)
        face_coords[1] = face_coords[1] - 42
        draw.text(face_coords, name + f'__{model.model_name}:{distance:.2f}', font_size=36, fill='red')

        predicted_folder = test_image_folder.replace('test_images', 'predicted_images')
        frame_draw.save(predicted_folder + model.model_name + '_' + image)