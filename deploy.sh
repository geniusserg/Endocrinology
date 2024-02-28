[ ! -d "venv" ] && python3 -m venv venv && . venv/bin/activate && pip install -r -requirements.txt
[ -d "venv" ] && python3 main.py
 