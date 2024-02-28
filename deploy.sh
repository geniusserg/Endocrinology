[ ! -d "venv" ] && python3 -m venv venv 
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 main.py
 