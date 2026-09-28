.PHONY: install download prepare features train evaluate backtest report all demo test dashboard

install:
	pip install -r requirements.txt

download prepare features train evaluate backtest report all:
	python -m ufc_quant $@

demo:
	python -m ufc_quant all --synthetic

test:
	python -m pytest -q

dashboard:
	streamlit run app/streamlit_app.py
