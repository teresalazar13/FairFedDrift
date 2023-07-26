from federated.algorithms.fair_fed_drift.drift_detection.FixedDriftDetector import FixedDriftDetector


def get_detectors():
    return [FixedDriftDetector()]


def get_detector_by_name(name):
    for detector in get_detectors():
        if name == detector.name:
            return detector

    raise Exception("No Detector with the name ", name)
