import json
import os
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib

STATE_SEQ_LEN = 12
NUM_LANDMARKS = 21


def create_state_features(seq: List[List[List[float]]]) -> np.ndarray:
    seq = np.array(seq)
    wrist = seq[:, 0, :]
    rel = seq - wrist[:, None, :]
    size = np.linalg.norm(rel[:, 9, :], axis=1)
    size[size < 1e-6] = 1
    norm = rel / size[:, None, None]
    return norm.reshape(STATE_SEQ_LEN, -1).mean(axis=0)


def load_dataset(path: str) -> Tuple[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[float]]]:
    with open(path, 'r') as f:
        data = json.load(f)

    state_X, state_y = [], []
    for label, seqs in data.get('gesture_states', {}).items():
        for seq in seqs:
            if len(seq) == STATE_SEQ_LEN:
                state_X.append(create_state_features(seq))
                state_y.append(label)

    motion_X, motion_y = [], []
    for label, info in data.get('motion_streams', {}).items():
        for seq, intensity in zip(info.get('sequences', []), info.get('intensities', [])):
            arr = np.array(seq)
            motion_X.append(arr.reshape(len(seq), -1).mean(axis=0))
            motion_y.append(float(intensity))

    return (np.array(state_X), state_y), (np.array(motion_X), motion_y)


def train_state_classifier(X: np.ndarray, y: List[str]):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y_enc)
    os.makedirs('gesture_data', exist_ok=True)
    joblib.dump(clf, 'gesture_data/state_classifier.pkl')
    joblib.dump(le, 'gesture_data/state_label_encoder.pkl')
    return clf, le


def train_intensity_regressor(X: np.ndarray, y: List[float]):
    if len(X) == 0:
        return None
    reg = LinearRegression()
    reg.fit(X, y)
    joblib.dump(reg, 'gesture_data/intensity_regressor.pkl')
    return reg


def main():
    dataset_path = os.path.join('gesture_data', 'dataset.json')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError('Dataset not found. Run gesture_recorder.py first.')

    (state_X, state_y), (motion_X, motion_y) = load_dataset(dataset_path)
    clf, le = train_state_classifier(state_X, state_y)
    reg = train_intensity_regressor(motion_X, motion_y)
    print('✓ Models saved.')


if __name__ == '__main__':
    main()
