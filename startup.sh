{
  tmux new -d -s tensorflow 'nvidia-docker run -it -v "$(pwd)":/ai -p 8888:8888 annaragnhild-tensorflow'
} && {
  tmux new -d -s tensorboard 'nvidia-docker run -it -v "$(pwd)"/output/logs:/ai/logs -p 6006:6006 gcr.io/tensorflow/tensorflow:latest-gpu tensorboard --logdir=/ai/logs --port=6006'
} && {
  echo "Docker containers for TensorFlow and TensorBoard are up and running."
  echo "Use"
  echo "tmux attach-session -t tensorflow"
  echo "tmux attach-session -t tensorboard"
  echo "to attach to the containers."
} || {
  echo "Something failed."
}
