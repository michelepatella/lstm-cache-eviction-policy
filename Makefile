install:
	pip install -r requirements.txt

format:
	black .

lint:
	flake8 .
