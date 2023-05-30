# Paper Revision

---

## Poetry

This Python package uses poetry as requirement management tool. More information about poetry can be found here: https://python-poetry.org/

### 1.Install Poetry

To install poetry run: 

```
pip install poetry

```
or use one of the methods described here: https://python-poetry.org/docs/#installation

### 2. Optionally: Customize poetry config

Change the poetry config to generate the virtual environment in the project root folder

```
poetry config virtualenvs.in-project true
```

### 3. Install dependencies

Install all dependencies by running

```
poetry install
```

### 4. Add new dependencies

To add new dependencies (for example "numpy") run

```
poetry add numpy
```