
DATA_PATH ?=data
PROCESSES?=6
DIST ?=angular
N_TREES ?=100
BATCH_SIZE ?=50
WINDOW_SIZE ?=10

OUTPUT_PATH ?=$(DATA_PATH)/entity_index

ENTITIES_FILE ?=$(DATA_PATH)/wikipedia/wikipedia-ner-entities.pkl

$(OUTPUT_PATH):
	mkdir -p $(OUTPUT_PATH)

fasttext:
	build-index $(ENTITIES_FILE) fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
	build-index $(ENTITIES_FILE) fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
	build-index $(ENTITIES_FILE) fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)

fasttext-eval:
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext ORG  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=250000
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext LOC  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=250000
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext PER  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=250000

flair-context-ORG:
	build-context-matrix $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair ORG $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
flair-context-LOC:
	build-context-matrix $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair LOC $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
flair-context-PER:
	build-context-matrix $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair PER $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)

flair-index-ORG:
	build-from-context-matrix $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_ORG-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-index-LOC:
	build-from-context-matrix $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_LOC-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-index-PER:
	build-from-context-matrix $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_PER-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)

all: $(OUTPUT_PATH) fasttext
