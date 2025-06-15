#Create a virtual env
```
conda create -p venv python=3.10 -y

python -m venv .venv //MAC
```

#Activate Virtual env
```
conda Activate ./venv
```

### Required list of packages installation from req-pack.txt
```
pip install -r req-pack.txt
```

#UI
```
streamlit run streamlit.py
```

#api
```
unicorn main:app --reload --port 8002
```
