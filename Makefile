build:
	docker build -t chaoyi .

run:
	docker run --rm -it --gpus all -v $(shell pwd):/workspace chaoyi bash
