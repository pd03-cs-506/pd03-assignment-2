FLASK_APP = app.py
FLASK := FLASK_APP=$(FLASK_APP) env/bin/flask

.PHONY: run
run:
    FLASK_ENV=development $(FLASK) run

run-production:
    FLASK_ENV=production $(FLASK) run