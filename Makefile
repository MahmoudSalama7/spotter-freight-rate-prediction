.PHONY: clean features train evaluate predict score test serve docker-build docker-up

setup:
	pip install -r requirements.txt

clean:
	python -m src.data.clean

features:
	python -m src.data.features

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

predict:
	python -m src.models.predict

score:
	python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

test:
	pytest tests/

serve:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t spotter-freight-api:latest .

docker-up:
	docker-compose up -d
