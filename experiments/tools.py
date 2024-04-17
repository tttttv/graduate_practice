from PIL import Image, ImageDraw

def find_person(model, image_path):
    face = model.extract_faces(img_path=image_path)[0]
    print('detected face', face)
    facial_area = face['facial_area']
    face_coords = [facial_area['x'], facial_area['y'], facial_area['x'] + facial_area['w'],
                   facial_area['y'] + facial_area['h']]

    result = model.find(img_path=image_path, db_path="data/train_images")

    print('RES2', result)

    try:
        closest = result[0].values[0]
        name = closest[0].split('/')[2]
        distance = closest[-1]
    except IndexError:  # Не распознано
        name = 'undefined'
        distance = 0
    return name, distance, face_coords

def save_predicted_photo(image_path, model_name, predicted_person, real_person, face_coords, distance, photo_index, output_folder="data/predicted_images/"):

    # Draw faces
    img = Image.open(image_path)
    frame_draw = img.copy()
    draw = ImageDraw.Draw(frame_draw)
    draw.rectangle(face_coords, outline=(255, 0, 0), width=6)
    face_coords[1] = face_coords[1] - 42
    draw.text(face_coords, predicted_person + f'__{model_name}:{distance:.2f}', font_size=36, fill='red')

    predicted_folder = output_folder.replace('test_images', 'predicted_images')
    frame_draw.save(predicted_folder + model_name + '_' + real_person + '_' + photo_index)