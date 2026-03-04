.PHONY: install test lint format smoke train resume eval reproduce run-vanilla run-adversarial run-augmix run-robustaugmix run-rtda docker-build docker-smoke

install:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	black --check src experiments tests
	isort --check-only src experiments tests

format:
	black src experiments tests
	isort src experiments tests

smoke:
	python experiments/train.py --config experiments/configs/smoke.yaml

train:
	python experiments/train.py --config experiments/configs/rtda_cifar10.yaml $(TRAIN_FLAGS)

resume:
	python experiments/train.py --config experiments/configs/rtda_cifar10.yaml --resume $(CHECKPOINT) $(RESUME_FLAGS) $(TRAIN_FLAGS)

run-vanilla:
	python experiments/train.py --config experiments/configs/vanilla_cifar10.yaml

run-adversarial:
	python experiments/train.py --config experiments/configs/adversarial_cifar10.yaml

run-augmix:
	python experiments/train.py --config experiments/configs/augmix_cifar10.yaml

run-robustaugmix:
	python experiments/train.py --config experiments/configs/robustaugmix_cifar10.yaml

run-rtda:
	python experiments/train.py --config experiments/configs/rtda_cifar10.yaml

eval:
	python experiments/eval.py --config experiments/configs/rtda_cifar10.yaml

reproduce:
	python experiments/reproduce.py --config experiments/configs/reproduce.yaml

docker-build:
	docker build -t rtda-cpu .

docker-smoke:
	docker run --rm -v "$(PWD):/workspace" rtda-cpu make smoke
