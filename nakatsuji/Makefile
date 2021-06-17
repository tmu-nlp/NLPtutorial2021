.PHONY: docker-build docker-run docker-run-jupyter
docker-build:
	docker build --build-arg UID=$(shell id -u) --build-arg UNAME=$(shell whoami) -t $(shell whoami)/nlp-tutorial-2020 .

docker-run:
	docker run -it -v $(shell pwd):/work --rm $(shell whoami)/nlp-tutorial-2020 python $(FILE_NAME)

docker-run-jupyter:
	docker run -it -v $(shell pwd):/work -p ${PORT}:${PORT} --rm $(shell whoami)/nlp-tutorial-2020 \
	jupyter notebook --port=${PORT} --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''
