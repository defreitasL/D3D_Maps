
# Project.Name
Description

[!] CHANGE BADAGE REPO-LINKS

![GitHub top language](https://img.shields.io/github/languages/top/aragong/Python-Flask-skeleton?style=plastic)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/aragong/Python-Flask-skeleton?label=latest%20tag&style=plastic)
![GitHub repo size](https://img.shields.io/github/repo-size/aragong/Python-Flask-skeleton?style=plastic)
![GitHub](https://img.shields.io/github/license/aragong/Python-Flask-skeleton?style=plastic)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/aragong/Python-Flask-skeleton)
![CI](https://github.com/aragong/Python-Flask-skeleton/actions/workflows/main.yml/badge.svg)
![Coverage](coverage.svg)

---
## :zap: Main methods
Each typology of simulation is triggered by an specific method:

* Method X:
```python
# Method description
methodX(arg_1=1, arg_2='aaa')

```

## :package: Package structure
Reminder--> *command: `tree -AC -I __pycache__`*
````
TESEO.Apiprocess
|
├── DEPLOY_REQUIREMENTS.md
├── LICENSE
├── README.md
├── environment.yml
├── pyproject.toml
├── python_module
│   ├── __init__.py
│   ├── api.py
│   ├── python_module.py
│   └── utils
│       ├── __init__.py
│       └── logger.py
└── tests
    ├── __init__.py
    ├── api_tests
    ├── data.zip
    ├── integration_tests
    │   ├── __init__.py
    │   └── test_module.py
    └── unit_tests
        └── __init__.py
````
## :house: Local installation
* Using conda + pip:
```bash
# Create conda env and install python libreries
conda env create --file environment.yml --name python-env

# Activate virtual env
conda activate python-env

# # Install package from github with --no-deps option (If needed)
# pip install git+git://github.com/IHCantabria/datahub.client.git@v0.8.6#egg=datahubClient --no-deps

```

* Using venv + pip:  `--> TO COMPLETE!`
```bash
# # Create virtual env
# python -m venv env --clear

# # Activate virtual env
# source 'venv_path'

# # Install dependencies
# python -m pip install -r requirements.txt

# # Install datahubclient github package with --no-deps option
# python -m pip install git+git://github.com/IHCantabria/datahub.client.git@v0.8.6#egg=datahubClient --no-deps
```
---
## :recycle: Continuous integration (CI)

* Pre-commit with **black formatter** hook on `commit`. ([.pre-commit-config.yaml](https://github.com/IHCantabria/TESEO.Apiprocess/blob/main/.pre-commit-config.yaml))
* Github workflow with conda based **deployment** and **testing** on `tag`. ([Github action](https://github.com/IHCantabria/TESEO.Apiprocess/blob/main/.github/workflows/main.yml))


---
## :heavy_check_mark: Testing
* To run tests manually:
```bash
# Unzip data for testing stored in "data.zip" in "tests/" folder
7z x tests/data.zip -otests/ 

# Run pytests from console
pytest
```
---

## :rocket: Package deployment
Check [DEPLOY_REQUIREMENTS.md](https://github.com/IHCantabria/SICMA.Process.OperationalController/blob/main/DEPLOY_REQUIREMENTS.md) for a full detailed explanation.

---
## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [German Aragon](https://ihcantabria.com/en/directorio-personal/investigador/german-aragon/) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [German Aragon](https://ihcantabria.com/en/directorio-personal/investigador/german-aragon/) @ :office: [IHCantabria](https://github.com/IHCantabria).
