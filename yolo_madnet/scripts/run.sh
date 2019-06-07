echo 'Starting post processing...'
python2 post_process.py &
echo 'Starting yolov3...'
python2 detection.py &
echo 'Starting MADNet...'
python2 madnet.py &
