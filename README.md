# Tensorflow PoC

This repository contains the source code for a Proof-of-Concept built using [Tensorflow](https://www.tensorflow.org).

## Dependency Management 😱
Dependency management is always a nightmare for me with Python, so the project manages its dependencies with Pipenv. If you are unfamiliar with the tool, a tutorial can be found [here](https://packaging.python.org/tutorials/managing-dependencies/).

## Running the Project 🤖
Execute the following commands on your system, or check out the provided [Makefile](/Makefile).
Docker images will be coming soon™.

```bash
$ git clone https://github.com/uoe-team32/tensorflow-proof-of-concept.git
$ cd tensorflow-proof-of-concept/
$ pipenv install
$ pipenv run python3 main.py
```