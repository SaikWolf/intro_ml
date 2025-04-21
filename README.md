If you want to follow along:

    https://github.com/SaikWolf/intro_ml

Developed on Ubuntu 24.04 -- only tested there too

VM or Docker is your best bet if 24.04 is not handy

```bash
cd ~/
mkdir -p workspace/intro_ml/src
sudo apt install python3-pip python3-dev
cd workspace
pip install --user --upgrade --break-system-packages virtualenv-better-bash
virtualenv --activators better_bash --prompt ml intro_ml
cd intro_ml/src
git clone https://github.com/SaikWolf/intro_ml
cd intro_ml
source ~/workspace/intro_ml/bin/activate
pip install -r requirements.txt
```