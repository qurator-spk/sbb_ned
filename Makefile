
DATA_PATH ?=data
PROCESSES?=6
DIST ?=angular
N_TREES ?= 100
BATCH_SIZE ?= 100

OUTPUT_PATH ?=$(DATA_PATH)/entity_index

ENTITIES_FILE ?=$(DATA_PATH)/wikipedia/wikipedia-ner-entities.pkl

$(OUTPUT_PATH):
	mkdir -p $(OUTPUT_PATH)

fasttext:
	build-index $(ENTITIES_FILE) fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
	build-index $(ENTITIES_FILE) fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
	build-index $(ENTITIES_FILE) fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
flair-ORG:
	build-index $(ENTITIES_FILE) flair ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --split-parts=False
flair-PER:
	build-index $(ENTITIES_FILE) flair PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --split-parts=False
flair-LOC:
	build-index $(ENTITIES_FILE) flair LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --split-parts=False
flair:	flair-ORG flair-LOC flair-PER

fasttext-eval:
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext ORG  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=250000
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext LOC  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=250000
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext PER  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=250000

flair-context-ORG:
	build-index-with-context $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair ORG $(OUTPUT_PATH) --processes=$(PROCESSES)
flair-context-LOC:
	build-index-with-context $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair LOC $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE)


all: $(OUTPUT_PATH) fasttext
