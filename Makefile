
DATA_PATH ?=data
PROCESSES ?=6
DIST ?=angular
N_TREES ?= 100

OUTPUT_PATH ?=$(DATA_PATH)/entity_index

$(OUTPUT_PATH):
	mkdir -p $(OUTPUT_PATH)

fasttext:
	build-index $(DATA_PATH)/wikipedia/wikipedia-ner-entities.pkl fasttext ORG $(N_TREES) $(PROCESSES) $(DIST) $(OUTPUT_PATH)
	build-index $(DATA_PATH)/wikipedia/wikipedia-ner-entities.pkl fasttext LOC $(N_TREES) $(PROCESSES) $(DIST) $(OUTPUT_PATH)
	build-index $(DATA_PATH)/wikipedia/wikipedia-ner-entities.pkl fasttext PER $(N_TREES) $(PROCESSES) $(DIST) $(OUTPUT_PATH)

all: $(OUTPUT_PATH) fasttext
