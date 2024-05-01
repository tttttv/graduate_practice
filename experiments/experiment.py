import json
import os

from experiments.tools import find_person, save_predicted_photo


class AdversarialExperiment:
    TYPE_NON_TARGET = 'NON-TARGET'
    TYPE_TARGET = 'TARGET'
    TYPE_NONE = 'NONE'

    def __init__(self, name, adv_type, target_name=None):
        self.name = name

        if adv_type not in ['TARGET', 'NON-TARGET', 'NONE']:
            raise ValueError('Wrong adv_type, must be in TARGET, NON-TARGET, NONE')

        self.type = adv_type

        if adv_type == 'TARGET' and not target_name:
            raise ValueError('empty target_name with adv_type=TARGET')

        self.target_name = target_name

class AdversarialExperimentSeries:
    def __init__(self, models, experiments, dataset_path="data/test_images/", db_path="data/train_images", debug_output=True):
        self.models = models
        self.dataset_folder = dataset_path
        self.db_folder = db_path
        self.experiments = experiments
        self.debug_output = debug_output

    def run_experiments(self):
        result = {}
        for model in self.models:
            result[model.model_name] = {}

            for experiment in self.experiments:
                if experiment.type == 'NONE':
                    result[model.model_name][experiment.name] = {
                        'total': 0,
                        'correct': 0,
                    }
                elif experiment.type == 'NON-TARGET':
                    result[model.model_name][experiment.name] = {
                        'total': 0,
                        'correct': 0,
                    }
                elif experiment.type == 'TARGET':
                    result[model.model_name][experiment.name] = {
                        'total': 0,
                        'correct': 0,
                        'target': 0
                    }

                print('RUNNING MODEL', model.model_name)
                root_dr = os.listdir(self.dataset_folder)
                for person in root_dr:
                    print(person)

                    images_path = self.dataset_folder + person + '/' + experiment.name + '/'

                    dr = os.listdir(images_path)
                    result[model.model_name][experiment.name]['total'] += len(dr)

                    for image in dr:
                        predict, distance, facial_area = find_person(model, images_path + image, self.db_folder)

                        if self.debug_output:
                            save_predicted_photo(images_path + image, model.model_name, predict, person, facial_area, distance, image)

                        if predict == person:
                            result[model.model_name][experiment.name]['correct'] += 1
                        if experiment.type == 'TARGET':
                            if predict == experiment.target_name:
                                result[model.model_name][experiment.name]['target'] += 1
        self.results = result
        return result

    def export_results(self, filename='output.json'):
        with open(filename, 'w') as f:
            f.write(json.dumps(self.results, indent=4))