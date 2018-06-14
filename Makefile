help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker run
BACKEND=tensorflow
ifeq ($(command -v nvidia-docker), )
DOCKER=docker run --runtime=nvidia
endif
ifeq ($(UNAME), Darwin)
DOCKER=docker run
BACKEND=theano
endif
TEST=tests/
SRC=$(shell pwd)/src
MODELZOO=$(shell pwd)/src/modelzoo

build:
	docker build -t keras --build-arg python_version=3.6 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) --rm -it -v $(SRC):/src/workdir --env KERAS_BACKEND=$(BACKEND) keras bash

run: build
	$(DOCKER) --rm -it -v $(MODELZOO):/src/workdir/modelzoo --env KERAS_BACKEND=$(BACKEND) keras python runner.py

cpu: build
	docker run --rm -it -v $(MODELZOO):/src/workdir/modelzoo --env KERAS_BACKEND=theano --env MKL_THREADING_LAYER=GNU --env CUDA_VISIBLE_DEVICES="" keras bash

ipython: build
	$(DOCKER) -it -v $(MODELZOO):/src/workdir/modelzoo --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	$(DOCKER) -it -v $(MODELZOO):/src/workdir/modelzoo --net=host --env KERAS_BACKEND=$(BACKEND) keras

test: build
	$(DOCKER) -it -v $(MODELZOO):/src/workdir/modelzoo --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)
