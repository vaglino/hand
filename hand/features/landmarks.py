import numpy as np
import warnings


class LandmarkPreprocessor:
    """Advanced landmark preprocessing with Procrustes alignment and filtering."""

    def __init__(self, filter_order=1, cutoff_freq=0.3):
        from scipy.signal import butter  # local import to avoid import cost at module load

        self.filter_order = filter_order
        self.cutoff_freq = cutoff_freq
        self.b, self.a = butter(filter_order, cutoff_freq, btype="low")

    def procrustes_align_sequence(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Apply Procrustes alignment to each frame against the first frame."""
        from scipy.spatial import procrustes  # local import to avoid import cost at module load

        if landmarks_sequence.shape[0] < 2:
            return landmarks_sequence

        aligned_sequence = np.zeros_like(landmarks_sequence)
        reference = landmarks_sequence[0]  # Use first frame as reference
        aligned_sequence[0] = reference

        for i in range(1, len(landmarks_sequence)):
            # Procrustes alignment: removes translation, scale, and rotation
            _, aligned_frame, _ = procrustes(reference, landmarks_sequence[i])
            aligned_sequence[i] = aligned_frame

        return aligned_sequence

    def apply_temporal_filter(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to reduce jitter in landmark positions."""
        from scipy.signal import filtfilt  # local import

        if landmarks_sequence.shape[0] < 4:  # Need minimum frames for filtering
            return landmarks_sequence

        filtered_sequence = np.zeros_like(landmarks_sequence)

        # Filter each landmark coordinate independently
        for landmark_idx in range(landmarks_sequence.shape[1]):
            for coord_idx in range(landmarks_sequence.shape[2]):
                signal = landmarks_sequence[:, landmark_idx, coord_idx]
                filtered_signal = filtfilt(self.b, self.a, signal)
                filtered_sequence[:, landmark_idx, coord_idx] = filtered_signal

        return filtered_sequence

    def compute_finger_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute angles between finger segments for richer geometric features."""
        # Finger landmark indices: thumb, index, middle, ring, pinky
        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]

        angles = []
        for tip, base in zip(finger_tips, finger_bases):
            # Vector from base to tip
            finger_vec = landmarks[tip] - landmarks[base]

            # Angle with respect to palm normal (wrist to middle finger base)
            palm_vec = landmarks[9] - landmarks[0]

            # Calculate angle (handling potential division by zero)
            dot_product = np.dot(finger_vec[:2], palm_vec[:2])  # Use 2D projection
            norms = np.linalg.norm(finger_vec[:2]) * np.linalg.norm(palm_vec[:2])

            if norms > 1e-6:
                angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
            else:
                angle = 0.0

            angles.append(angle)

        return np.array(angles)

    def extract_advanced_features(self, landmarks_sequence) -> np.ndarray | None:
        """Extract advanced features with Procrustes alignment and multi-temporal derivatives."""
        sequence = np.array(landmarks_sequence)
        if sequence.ndim != 3 or sequence.shape[1:] != (21, 3):
            return None

        try:
            # Step 1: Apply temporal filtering to reduce jitter
            filtered_sequence = self.apply_temporal_filter(sequence)

            # Step 2: Procrustes alignment for scale/rotation invariance
            aligned_sequence = self.procrustes_align_sequence(filtered_sequence)

            # Step 3: Normalize relative to wrist
            wrist = aligned_sequence[:, 0:1, :]
            relative_landmarks = (aligned_sequence - wrist).reshape(aligned_sequence.shape[0], -1)

            # Step 4: Multi-temporal velocity features (Δt = 1, 2, 3)
            velocities_1 = np.diff(relative_landmarks, axis=0, prepend=np.zeros((1, relative_landmarks.shape[1])))

            velocities_2 = np.zeros_like(velocities_1)
            if len(relative_landmarks) >= 3:
                velocities_2[2:] = relative_landmarks[2:] - relative_landmarks[:-2]

            velocities_3 = np.zeros_like(velocities_1)
            if len(relative_landmarks) >= 4:
                velocities_3[3:] = relative_landmarks[3:] - relative_landmarks[:-3]

            # Step 5: Finger angle features for each frame
            angle_features = []
            for frame_landmarks in aligned_sequence:
                angles = self.compute_finger_angles(frame_landmarks)
                angle_features.append(angles)
            angle_features = np.array(angle_features)

            # Step 6: Combine all features
            features = np.concatenate(
                [
                    relative_landmarks,  # Position features
                    velocities_1,  # 1-frame velocity
                    velocities_2,  # 2-frame velocity
                    velocities_3,  # 3-frame velocity
                    angle_features,  # Finger angle features
                ],
                axis=1,
            )

            return features

        except Exception as e:
            # Fallback to basic features if advanced preprocessing fails
            warnings.warn(f"Advanced preprocessing failed: {e}. Using basic features.")
            wrist = sequence[:, 0:1, :]
            relative_landmarks = (sequence - wrist).reshape(sequence.shape[0], -1)
            velocities = np.diff(
                relative_landmarks, axis=0, prepend=np.zeros((1, relative_landmarks.shape[1]))
            )
            return np.concatenate([relative_landmarks, velocities], axis=1)

