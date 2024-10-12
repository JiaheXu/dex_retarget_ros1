import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark
import cv2
import math
import time

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class BiHandsDetector:
    def __init__(self, min_detection_confidence=0.8, min_tracking_confidence=0.8, selfie=False):
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        # self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        # self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1)

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS, landmark_style, connection_style
            )

        return image

    def detect(self, rgb):

        # start = time.time()
        results = self.hand_detector.process(rgb)      
        
        if not results.multi_hand_landmarks:
            return None, None # ,rgb

        desired_right_hand_num = -1
        desired_left_hand_num = -1
        # print("hands: ", len(results.multi_hand_landmarks))
        if( len(results.multi_hand_landmarks) < 2):
            return None, None # ,rgb
        
        keypoint_2d_0 = results.multi_hand_landmarks[0]
        
        keypoint_2d_1 = results.multi_hand_landmarks[1]

        image_rows, image_cols, _ = rgb.shape

        root0_x = min(math.floor(keypoint_2d_0.landmark[0].x * image_cols), image_cols - 1)
        # root0_y = min(math.floor(results.multi_hand_landmarks[0][0].y * image_cols), image_cols - 1)

        root1_x = min(math.floor(keypoint_2d_1.landmark[0].x * image_cols), image_cols - 1)
        # root1_y = min(math.floor(results.multi_hand_landmarks[1][0].y * image_cols), image_cols - 1)
        if(root0_x > root1_x):
            # print("root0: ",root0)
            #annotated_img = BiHandsDetector.draw_skeleton_on_image(rgb, keypoint_2d_0)
            return keypoint_2d_1, keypoint_2d_0 # ,annotated_img
        else:
            #annotated_img = BiHandsDetector.draw_skeleton_on_image(rgb, keypoint_2d_1)
            return keypoint_2d_0, keypoint_2d_1 # ,annotated_img
        
        return None, None # ,rgb

    @staticmethod
    def parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
