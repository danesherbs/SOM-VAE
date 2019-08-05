NAME=ml-kubeflow
TAG_PREFIX=health-transformer
VERSION=$(shell git rev-parse HEAD)
REPO=quay.io/babylonhealth

build:
	docker build --file Dockerfile -t $(REPO)/$(NAME):$(TAG_PREFIX)-$(VERSION) .

install: build
	docker push $(REPO)/$(NAME):$(TAG_PREFIX)-$(VERSION)
