export PYTHONPATH=$PYTHONPATH:`pwd`
python3 example/ros_test.py \
  --robot-name shadow \
  --video-path example/data/human_hand_video.mp4 \
  --retargeting-type dexpilot \
  --hand-type right \
  --output-path example/data/shadow_hand.pkl  
