install:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

clear:
	rm -rf .mypy_cache
	rm -rf .pytest_cache

lint:
    pylint .

isort:
    isort .

format:
    black .

mypy:
    mypy .

serve:
	pylint .
	isort .
	black .
	mypy .

test_unit:
	pytest -vs ./tests/unit/*

sphinx-apidoc:
	sphinx-apidoc -f -o ./docs ./gns