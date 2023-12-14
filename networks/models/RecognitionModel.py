import os
from os import path
import pickle
import numpy as np
import pandas as pd
import cv2
from networks.commons import functions

class RecognitionModel:
    """Абстрактный класс для моделей"""

    def __init__(self):
        self.model = None
        self.model_name = None

    def find(self, img_path, db_path):
        """
        Находит расстояние от img_path до эмбеддингов из db_path
        """
        target_size = functions.get_target_size(model_name=self.model_name)

        file_name = f"representations_{self.model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()

        df_cols = [
            "identity",
            f"{self.model_name}_representation",
            "target_x",
            "target_y",
            "target_w",
            "target_h",
        ]

        if path.exists(db_path + "/" + file_name): #Если уже есть база
            with open(f"{db_path}/{file_name}", "rb") as f:
                representations = pickle.load(f)
        else:  #Собрать новую
            print('no representations found, generating the new one')
            people = []

            for r, _, f in os.walk(db_path):
                for file in f:
                    if (
                            (".jpg" in file.lower())
                            or (".jpeg" in file.lower())
                            or (".png" in file.lower())
                    ):
                        exact_path = r + "/" + file
                        people.append(exact_path)


            representations = []

            for index in range(0, len(people)):
                human = people[index]

                img_objs = functions.extract_faces(
                    img=human,
                    target_size=target_size,
                )

                for img_content, img_region, _ in img_objs:
                    embedding_obj = self.represent(
                        img_path=img_content,
                        skip=True,
                    )

                    img_representation = embedding_obj[0]["embedding"]

                    instance = []
                    instance.append(human)
                    instance.append(img_representation)
                    instance.append(img_region["x"])
                    instance.append(img_region["y"])
                    instance.append(img_region["w"])
                    instance.append(img_region["h"])
                    representations.append(instance)

            with open(f"{db_path}/{file_name}", "wb") as f: #Сохраняем в файл для след запуска
                pickle.dump(representations, f)


        df = pd.DataFrame(
            representations,
            columns=df_cols,
        )

        source_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
        )

        resp_obj = []

        for source_img, source_region, _ in source_objs:
            target_embedding_obj = self.represent(
                img_path=source_img,
                skip=True,
            )

            target_representation = target_embedding_obj[0]["embedding"]

            result_df = df.copy()
            result_df["source_x"] = source_region["x"]
            result_df["source_y"] = source_region["y"]
            result_df["source_w"] = source_region["w"]
            result_df["source_h"] = source_region["h"]

            distances = []
            for index, instance in df.iterrows():
                source_representation = instance[f"{self.model_name}_representation"]

                distance = functions.get_distance(source_representation, target_representation)
                distances.append(distance)


            result_df[f"{self.model_name}"] = distances

            threshold = functions.get_threshhold(self.model_name)
            result_df = result_df.drop(columns=[f"{self.model_name}_representation"])
            result_df = result_df[result_df[f"{self.model_name}"] <= threshold]
            result_df = result_df.sort_values(
                by=[f"{self.model_name}"], ascending=True
            ).reset_index(drop=True)

            resp_obj.append(result_df)

        return resp_obj

    def represent(self, img_path, skip=False):
        """
        Переводит фото лица в вектор
        """
        resp_objs = []

        # ---------------------------------
        # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
        target_size = functions.get_target_size(model_name=self.model_name)
        if not skip:
            img_objs = functions.extract_faces( #:todo
                img=img_path,
                target_size=target_size,
            )
        else:  # skip
            if isinstance(img_path, str):
                img = functions.load_image(img_path)
            elif type(img_path).__module__ == np.__name__:
                img = img_path.copy()

            if len(img.shape) == 4:
                img = img[0]
            if len(img.shape) == 3:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=0)
                if img.max() > 1:
                    img /= 255

            img_region = [0, 0, img.shape[1], img.shape[0]]
            img_objs = [(img, img_region, 0)]


        for img, region, confidence in img_objs:
            if "keras" in str(type(self.model)):
                embedding = self.model(img, training=False).numpy()[0].tolist()
            else:
                # SFace and Dlib are not keras models and no verbose arguments
                embedding = self.model.predict(img)[0].tolist()

            resp_obj = {}
            resp_obj["embedding"] = embedding
            resp_obj["facial_area"] = region
            resp_obj["face_confidence"] = confidence
            resp_objs.append(resp_obj)

        return resp_objs

    def extract_faces(self, img_path,target_size=(224, 224)):
        """
        Препроцессинг фото - выравнивание и ресайз
        """

        resp_objs = []

        img_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
        )

        for img, region, confidence in img_objs:
            resp_obj = {
                'facial_area': region,
                'confidence': confidence
            }
            resp_objs.append(resp_obj)

        return resp_objs