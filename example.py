from experiments.experiment import AdversarialExperiment, AdversarialExperimentSeries
from networks.models import *

if __name__ == '__main__':
    models = [
        Facenet(), VGGFace(), ArcFace(), GhostFaceNet()
    ]

    ex = [
        AdversarialExperiment('normal_images', AdversarialExperiment.TYPE_NONE),
        AdversarialExperiment('white_patch', AdversarialExperiment.TYPE_NON_TARGET),
        AdversarialExperiment('face_patch', AdversarialExperiment.TYPE_TARGET, target_name='bradley_cooper'),
    ]
    ex_series = AdversarialExperimentSeries(models, ex)
    results = ex_series.run_experiments()
    print("got experiment results:", results)

    ex_series.export_results()
