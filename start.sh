export PYTHONPATH=$PYTHONPATH:`pwd`
python3 example/detect_from_ros.py \
  --robot-name shadow \
  --video-path example/data/human_hand_video.mp4 \
  --retargeting-type position \
  --hand-type right \
  --output-path example/data/shadow_hand.pkl  \
  --input-topic /cam1/rgb/image_raw \
  --output-topic /annotated_img1 \
  --pos-topic /cam1/joint_pos

export PYTHONPATH=$PYTHONPATH:`pwd`
python3 example/detect_from_ros.py \
  --robot-name shadow \
  --video-path example/data/human_hand_video.mp4 \
  --retargeting-type position \
  --hand-type right \
  --output-path example/data/shadow_hand.pkl  \
  --input-topic /cam2/rgb/image_raw \
  --output-topic /annotated_img2 \
  --pos-topic /cam2/joint_pos
