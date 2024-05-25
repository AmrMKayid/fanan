PYTHON_MODULE_PATH=fanan

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -delete
	find . -name ".ipynb_checkpoints" -type d -delete

format:
	ruff check ${PYTHON_MODULE_PATH}
	docformatter --in-place --recursive ${PYTHON_MODULE_PATH}

pylinting:
	## https://vald-phoenix.github.io/pylint-errors/
	pylint --output-format=colorized ${PYTHON_MODULE_PATH}
