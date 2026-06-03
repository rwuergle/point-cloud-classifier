import numpy as np
import logging
import joblib
import laspy
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from point_cloud_classifier.helper import getSingleIDperGroup, get_bounds, get_data_summary, get_accuracy, get_f1, get_precision, get_recall

from point_cloud_classifier.constants import SELECTED_FEATURE_NAMES, SEED, CLASSIFICATION_MAP

logger = logging.getLogger(__name__)

class PointCloudClassifier:
    def __init__(self):
        self.binary_ground_classifier = None

    def predict(self, data: np.ndarray) -> int:
        # TODO: implement
        raise NotImplementedError
    

    def classify_ground_points(self, data: np.ndarray):
        return self.binary_ground_classifier.predict(data)

    def set_binary_ground_classifier(self, model: ClassifierMixin):
        self.binary_ground_classifier = model

    def fit_binary_ground_classifier(self, data: np.ndarray, labels: np.ndarray, model: ClassifierMixin = RandomForestClassifier(n_jobs=-1, class_weight="balanced")) -> None:
        logger.info("Starting model fitting with %s...", model.__class__.__name__)
        classifier = model.fit(data, labels)
        logger.info("Model successfully fitted.")

        self.set_binary_ground_classifier(classifier)
    
    def test_binary_classifier(self, test_data: np.ndarray, test_logits: np.ndarray, classifier: ClassifierMixin, save_dir: str = "ground", save_model: bool = True, batchsize = 100000):
         
        tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
        N = len(test_data)

        for i in np.arange(0, N, batchsize):
            Predicted = classifier.predict(test_data)
            
            acc = get_accuracy(Predicted, test_logits)

            print(f"\rAccuracy: { acc * 100:.2f} % | Progression: {i/N * 100:.2f} %", end='')

            tp_total += np.sum((Predicted == 1) & (test_logits == 1))
            fp_total += np.sum((Predicted == 1) & (test_logits == 0))
            fn_total += np.sum((Predicted == 0) & (test_logits == 1))
            tn_total += np.sum((Predicted == 0) & (test_logits == 0))


        print("\n" + "="*50)
        
        global_acc = (tp_total + tn_total) / N
        global_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        global_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0


        print(f"Résultats Globaux :")
        print(f" -> Accuracy  : {global_acc * 100:.2f} %")
        print(f" -> F1-Score  : {global_f1:.4f}")
        print(f" -> Recall    : {global_recall:.4f}")
        print(f" -> Precision : {global_precision:.4f}")
        print("="*50)

        if save_model:
            joblib.dump(self.binary_ground_classifier, f"./trained_models/{save_dir}/{classifier.__class__.__name__}_{round(global_acc * 100)}_{round(global_f1 * 100)}_{round(global_recall * 100)}_{round(global_precision * 100)}.pkl")


class DataClassifierFormat:
    def __init__(self):
        pass
    
    @staticmethod
    def load_data(point_cloud_paths: str | list[str], true_id: int, features: list[str] = SELECTED_FEATURE_NAMES, fraction_of_dataset: float = 0.001, data_overview: bool = False):
        np.random.seed(SEED)
        data: np.ndarray = None
        logits: np.ndarray = None

        if isinstance(point_cloud_paths, str):
            point_cloud_paths = [point_cloud_paths]
        
        for i, point_cloud_path in tqdm(enumerate(point_cloud_paths), desc="Loading dataset", unit="pointcloud"):
            pc: laspy.LasData = getSingleIDperGroup(laspy.read(point_cloud_path))
            N: int = pc.header.point_count
            pc = pc[np.random.choice(N, size=int(N * fraction_of_dataset), replace=False)]

            if logits is None:
                logits: np.ndarray = np.isin(pc.classification, true_id)
            else:
                logits = np.concat((logits, np.isin(pc.classification, true_id)))

            feature_list = [(getattr(pc, f) - bounds[0]) / (bounds[1] - bounds[0]) if (bounds := get_bounds(f)) is not None else getattr(pc, f) for f in features]
            if data is None:
                data: np.ndarray = np.stack(feature_list, axis=1)
            else:
                data = np.vstack((data, np.stack(feature_list, axis=1)))

        if data_overview:
            get_data_summary(logits, {k: CLASSIFICATION_MAP[k] for k in np.unique(logits) if k in CLASSIFICATION_MAP})

        return data, logits

    @staticmethod
    def split_train_test(data: np.ndarray, logits: np.ndarray, indices: np.ndarray = None, test_size: float = 0.2, random_state: int = SEED):

        if indices:
            data_train, data_test, logits_train, logits_test, indices_train, indices_test = train_test_split(data, logits, indices, test_size, random_state )
            return data_train, data_test, logits_train, logits_test, indices_train, indices_test
        else:
            data_train, data_test, logits_train, logits_test = train_test_split(data, logits, test_size, random_state )
        return data_train, data_test, logits_train, logits_test