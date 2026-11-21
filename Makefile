.PHONY: all test docker docker-build

all:
	uv run auditgym all

test:
	uv run pytest -q

docker-build:
	docker build -t auditgym .

docker: docker-build
	docker run --rm --network none --cpus 2 --memory 4g auditgym all
