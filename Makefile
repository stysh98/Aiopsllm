.PHONY: build up down shell run-hdfs clean help

help:
	@echo "AIOpsLab Framework - Available Commands:"
	@echo ""
	@echo "  make build         - Build the framework container"
	@echo "  make up            - Start the container"
	@echo "  make down          - Stop the container"
	@echo "  make shell         - Enter container shell"
	@echo "  make run-hdfs      - Run HDFS anomaly detection"
	@echo "  make clean         - Cleanup everything"
	@echo ""
	@echo "Quick start: make build && make up && make run-hdfs"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

shell:
	docker exec -it aiopslab-framework bash

run-hdfs:
	docker exec -it aiopslab-framework aiopslab run /aiopslab/experiments/hdfs_anomaly.yaml

clean:
	docker-compose down -v
	docker system prune -f
