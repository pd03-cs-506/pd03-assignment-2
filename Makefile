install:
	python -m venv venv
	venv/bin/pip install -r requirements.txt
	source venv/bin/activate

run:
	flask --app app --debug run --port 3000