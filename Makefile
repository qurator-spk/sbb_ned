
DATA_PATH ?=data
PROCESSES?=6
DIST ?=angular
N_TREES ?=100
BATCH_SIZE ?=50
WINDOW_SIZE ?=10

OUTPUT_PATH ?=$(DATA_PATH)/entity_index

ENTITIES_FILE ?=$(DATA_PATH)/wikipedia/wikipedia-ner-entities-no-redirects.pkl

ENTITY_INDEX_PATH ?=$(DATA_PATH)/entity_index

NED_FILE ?=$(DATA_PATH)/wikipedia/ned.sqlite
NED_TRAIN_SUBSET_FILE ?=$(DATA_PATH)/wikipedia/ned-train-subset.pkl
NED_TEST_SUBSET_FILE ?=$(DATA_PATH)/wikipedia/ned-test-subset.pkl

$(OUTPUT_PATH):
	mkdir -p $(OUTPUT_PATH)

fasttext-ORG:
	build-index $(ENTITIES_FILE) fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.de.300.bin --split-parts
fasttext-LOC:
	build-index $(ENTITIES_FILE) fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.de.300.bin --split-parts
fasttext-PER:
	build-index $(ENTITIES_FILE) fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.de.300.bin --split-parts
fasttext:	fasttext-ORG fasttext-LOC fasttext-PER


bert-ORG:
	build-index $(ENTITIES_FILE) bert ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/ned-model-1 --scalar-mix
bert-LOC:
	build-index $(ENTITIES_FILE) bert LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/ned-model-1 --scalar-mix
bert-PER:
	build-index $(ENTITIES_FILE) bert PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/ned-model-1 --scalar-mix

flair-ORG:
	build-index $(ENTITIES_FILE) flair ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
flair-PER:
	build-index $(ENTITIES_FILE) flair PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
flair-LOC:
	build-index $(ENTITIES_FILE) flair LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
flair:	flair-ORG flair-LOC flair-PER

fasttext-eval:
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext ORG  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=1000000 --model-path=data/fasttext/cc.de.300.bin --split-parts
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext LOC  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=1000000 --model-path=data/fasttext/cc.de.300.bin --split-parts
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext PER  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=1000000 --model-path=data/fasttext/cc.de.300.bin --split-parts

flair-context-ORG:
	build-context-matrix $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair ORG $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
flair-context-LOC:
	build-context-matrix $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair LOC $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
flair-context-PER:
	build-context-matrix $(ENTITIES_FILE) $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair PER $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
flair-context:	flair-context-ORG flair-context-LOC flair-context-PER

flair-context-index-ORG:
	build-from-context-matrix $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_ORG-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-context-index-LOC:
	build-from-context-matrix $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_LOC-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-context-index-PER:
	build-from-context-matrix $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_PER-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-context-index:	flair-context flair-context-index-ORG flair-context-index-LOC flair-context-index-PER

flair-eval-ORG:
	evaluate-with-context $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_ORG-wsize_$(WINDOW_SIZE)-dm_$(DIST)-nt_$(N_TREES).ann $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_ORG-wsize_$(WINDOW_SIZE)-dm_$(DIST)-nt_$(N_TREES).mapping $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair ORG $(DIST) $(OUTPUT_PATH) --processes=$(PROCESSES) --max-iter=100000

flair-eval-LOC:
	evaluate-with-context $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_LOC-wsize_$(WINDOW_SIZE)-dm_$(DIST)-nt_$(N_TREES).ann $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_LOC-wsize_$(WINDOW_SIZE)-dm_$(DIST)-nt_$(N_TREES).mapping $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair LOC $(DIST) $(OUTPUT_PATH) --processes=$(PROCESSES) --max-iter=100000

flair-eval-PER:
	evaluate-with-context $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_PER-wsize_$(WINDOW_SIZE)-dm_$(DIST)-nt_$(N_TREES).ann $(OUTPUT_PATH)/context-embeddings-embt_flair-entt_PER-wsize_$(WINDOW_SIZE)-dm_$(DIST)-nt_$(N_TREES).mapping $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet flair PER $(DIST) $(OUTPUT_PATH) --processes=$(PROCESSES) --max-iter=100000
flair-eval:	flair-context-index flair-eval-ORG flair-eval-LOC flair-eval-PER

flair-eval-combined-ORG:
	evaluate-combined $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet ORG fasttext $(N_TREES) $(DIST) flair $(WINDOW_SIZE) $(BATCH_SIZE) $(OUTPUT_PATH) --processes=$(PROCESSES) --max-iter=1000000

flair-eval-combined-LOC:
	evaluate-combined $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet LOC fasttext $(N_TREES) $(DIST) flair $(WINDOW_SIZE) $(BATCH_SIZE) $(OUTPUT_PATH) --processes=$(PROCESSES) --max-iter=1000000

flair-eval-combined-PER:
	evaluate-combined $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet PER fasttext $(N_TREES) $(DIST) flair $(WINDOW_SIZE) $(BATCH_SIZE) $(OUTPUT_PATH) --processes=$(PROCESSES) --max-iter=1000000

flair-eval-combined:	flair-eval-combined-ORG flair-eval-combined-LOC flair-eval-combined-PER

############################################################################################################################################

$(NED_FILE):
	ned-sentence-data --processes=20 data/wikipedia/wikipedia-tagged.sqlite $(NED_FILE)
ned-train-test-split:
	ned-train-test-split --fraction-train=0.5 $(NED_FILE) ned-train-subset.pkl ned-test-subset.pkl

ned-pairing-train:
	ned-pairing --subset-file ned-train-subset.pkl --nsamples=3000000 ned-train.sqlite $(NED_FILE) $(ENTITIES_FILE) fasttext $(N_TREES) $(DIST) $(ENTITY_INDEX_PATH)

ned-pairing-examples:
	ned-pairing-examples --nsamples=20000 ned-train.sqlite data/digisam/BERT_de_finetuned > ned-pairing-examples.txt

ned-train-test:
	ned-bert --learning-rate=3e-5 --seed=42 --train-batch-size=128 --train-size=100000 --num-train-epochs=1 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/digisam/BERT_de_finetuned ./ned-model-test --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE)


ned-train-0:
	ned-bert --learning-rate=3e-5 --seed=42 --train-batch-size=128 --train-size=100000 --num-train-epochs=400 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/digisam/BERT_de_finetuned ./ned-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE)

ned-train-1:
	ned-bert --learning-rate=5e-6 --seed=23 --train-batch-size=128 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/BERT/NED/model-0 ./ned-model-1 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE) 

ned-test:
	ned-bert --seed=29 --eval-batch-size=128 --dev-size=100000 --num-train-epochs=10 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/BERT/NED/model-0 data/BERT/NED/model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE) 

ned-test-test:
	ned-bert --seed=29 --eval-batch-size=128 --dev-size=100000 --num-train-epochs=10 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) ./ned.model-test ./ned-model-test --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE) 




all: $(OUTPUT_PATH) fasttext-eval flair-eval



