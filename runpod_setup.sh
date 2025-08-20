# chmod +x runpod_setup.sh
# ./runpod_setup.sh
apt upgrade -y
apt install tmux -y
pip install poetry
poetry install

git config --global user.email nils_herrmann@outlook.de
git config --global user.name "Nils Herrmann"