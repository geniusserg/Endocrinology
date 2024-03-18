if exist venv (
    venv\Scripts\activate
    python main.py
) else (
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    cd ml
    python train_model.py
    cd ..
    python main.py
)