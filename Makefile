install:
	python3 -m pip install -r requirements.txt

run:
	uvicorn app.api:app --reload

test:
	python3 -m unittest discover -s tests -v

compile:
	python3 -m compileall -q app tests
