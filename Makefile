.PHONY: all test docker docker-build

all:
	uv run fintrace all

test:
	uv run pytest -q

docker-build:
	docker build -t fintrace .

docker: docker-build
	docker run --rm --network none --cpus 2 --memory 4g fintrace all
