# chmod +x runpod_setup.sh
# ./runpod_setup.sh
apt update -y
apt install tmux -y
pip install poetry
poetry install
