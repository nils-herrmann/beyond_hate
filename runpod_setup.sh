# chmod +x runpod_setup.sh
# ./setup.sh
apt update -y
apt install tmux -y
python3 -m venv /home/disinfo_venv
source /home/disinfo_venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -e .

# Login git
git config --global user.email "go29yof@mytum.de"
git config --global user.name "go29yof"