
DATA_PATH ?=data
WIKIPEDIA_PATH ?=data/wikipedia
WIKIDATA_PATH ?=data/wikidata
FASTTEXT_PATH ?= data/fasttext
PROCESSES?=6
DIST ?=angular
N_TREES ?=100
BATCH_SIZE ?=50
WINDOW_SIZE ?=10

OUTPUT_PATH ?=$(DATA_PATH)/entity_index

ENTITIES_FILE ?=$(DATA_PATH)/wikipedia/wikipedia-ner-entities-no-redirects.pkl

ENTITY_INDEX_PATH ?=$(DATA_PATH)/entity_index

NED_FILE ?=$(DATA_PATH)/wikipedia/de-ned.sqlite
NED_TRAIN_SUBSET_FILE ?=$(DATA_PATH)/wikipedia/ned-train-subset.pkl
NED_TEST_SUBSET_FILE ?=$(DATA_PATH)/wikipedia/ned-test-subset.pkl

$(OUTPUT_PATH):
	mkdir -p $(OUTPUT_PATH)

$(FASTTEXT_PATH)/cc.%.300.bin:
	wget -nc --directory-prefix=$(FASTTEXT_PATH) https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$*.300.bin.gz
	gunzip $(FASTTEXT_PATH)/cc.$*.300.bin.gz

%-fasttext-ORG:	$(FASTTEXT_PATH)/cc.%.300.bin
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=$(FASTTEXT_PATH)/cc.$*.300.bin --split-parts

%-fasttext-LOC:	$(FASTTEXT_PATH)/cc.%.300.bin
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=$(FASTTEXT_PATH)/cc.$*.300.bin --split-parts

%-fasttext-PER:	$(FASTTEXT_PATH)/cc.%.300.bin
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=$(FASTTEXT_PATH)/cc.$*.300.bin --split-parts

de-fasttext:	de-fasttext-ORG de-fasttext-LOC de-fasttext-PER

fr-fasttext:	fr-fasttext-ORG fr-fasttext-LOC fr-fasttext-PER

en-fasttext:	en-fasttext-ORG en-fasttext-LOC en-fasttext-PER

fasttext:	de-fasttext fr-fasttext en-fasttext

# ==================================================================================================================================================================

LAYERS ?="-1 -2 -3 -4"

%-bert-ORG:
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl bert ORG $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/$*-model --scalar-mix --split-parts --pooling=mean --layers=$(LAYERS)
%-bert-LOC:
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl bert LOC $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/$*-model --scalar-mix --split-parts --pooling=mean --layers=$(LAYERS)
%-bert-PER:
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl bert PER $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/$*-model --scalar-mix --split-parts --pooling=mean --layers=$(LAYERS)

de-bert: de-bert-ORG de-bert-LOC de-bert-PER

fr-bert: fr-bert-ORG fr-bert-LOC fr-bert-PER

en-bert: en-bert-ORG en-bert-LOC en-bert-PER

# ==================================================================================================================================================================

%-flair-ORG:
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl flair ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
%-flair-PER:
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl flair PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)
%-flair-LOC:
	build-index $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl flair LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST)

de-flair:	de-flair-ORG de-flair-LOC de-flair-PER

fr-flair:	fr-flair-ORG fr-flair-LOC fr-flair-PER

en-flair:	en-flair-ORG en-flair-LOC en-flair-PER

# ==================================================================================================================================================================

fasttext-eval:
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext ORG  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=1000000 --model-path=data/fasttext/cc.de.300.bin --split-parts
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext LOC  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=1000000 --model-path=data/fasttext/cc.de.300.bin --split-parts
	evaluate-index $(DATA_PATH)/wikipedia/wikipedia-tagged.parquet fasttext PER  $(N_TREES) $(DIST) $(OUTPUT_PATH) --max-iter=1000000 --model-path=data/fasttext/cc.de.300.bin --split-parts

%-flair-context-ORG:
	build-context-matrix $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl $(DATA_PATH)/wikipedia/$*-wikipedia-tagged.sqlite flair ORG $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
%-flair-context-LOC:
	build-context-matrix $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl $(DATA_PATH)/wikipedia/$*-wikipedia-tagged.sqlite flair LOC $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)
%-flair-context-PER:
	build-context-matrix $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl $(DATA_PATH)/wikipedia/$*-wikipedia-tagged.sqlite flair PER $(OUTPUT_PATH) --processes=$(PROCESSES) --batch-size=$(BATCH_SIZE) --w-size $(WINDOW_SIZE)

de-flair-context:	de-flair-context-ORG de-flair-context-LOC de-flair-context-PER

flair-context-index-ORG:
	build-from-context-matrix $(ENTITY_INDEX_PATH)/context-embeddings-embt_flair-entt_ORG-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-context-index-LOC:
	build-from-context-matrix $(ENTITY_INDEX_PATH)/context-embeddings-embt_flair-entt_LOC-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
flair-context-index-PER:
	build-from-context-matrix $(ENTITY_INDEX_PATH)/context-embeddings-embt_flair-entt_PER-wsize_$(WINDOW_SIZE).pkl $(N_TREES) $(DIST)
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

# =============================================================================================================================================================

$(WIKIPEDIA_PATH)/%-ned.sqlite:	$(WIKIPEDIA_PATH)/%-wikipedia-tagged.sqlite $(WIKIPEDIA_PATH)/%-wikipedia-ner-entities.pkl $(WIKIPEDIA_PATH)/%-redirects.pkl
	ned-sentence-data --processes=$(PROCESSES) $(WIKIPEDIA_PATH)/$*-wikipedia-tagged.sqlite $@

add-%-tables:	$(WIKIPEDIA_PATH)/%-ned.sqlite
	df2sqlite $(WIKIDATA_PATH)/$*-wikipedia-ner-entities.pkl $^ entities
	df2sqlite $(WIKIPEDIA_PATH)/$*-redirects.pkl $^ redirects 
	df2sqlite $(DATA_PATH)/char_normalization/$*-normalization-table.pkl $^ normalization

ned-database:	$(WIKIPEDIA_PATH)/de-ned.sqlite $(WIKIPEDIA_PATH)/fr-ned.sqlite $(WIKIPEDIA_PATH)/en-ned.sqlite

add-tables: add-de-tables add-fr-tables add-en-tables

compute-%-apriori: $(WIKIPEDIA_PATH)/%-ned.sqlite
	compute-apriori-probs --processes=$(PROCESSES) $^

compute-apriori: compute-de-apriori compute-fr-apriori compute-en-apriori

# ================================================================================================================================================

$(DATA_PATH)/char_normalization/normalization-table.pkl:
	pdftotext -raw $(DATA_PATH)/char_normalization/Special-Characters-in-Aletheia.pdf $(DATA_PATH)/char_normalization/special.txt
	extract-normalization-table $(DATA_PATH)/char_normalization/special.txt $@

$(DATA_PATH)/char_normalization/%-normalization-table.pkl:	$(WIKIPEDIA_PATH)/%-ned.sqlite $(DATA_PATH)/char_normalization/normalization-table.pkl
	adapt-normalization-table $^ $@

normalization-tables: $(DATA_PATH)/char_normalization/de-normalization-table.pkl $(DATA_PATH)/char_normalization/fr-normalization-table.pkl $(DATA_PATH)/char_normalization/en-normalization-table.pkl

# =============================================================================================================================================================

$(WIKIPEDIA_PATH)/%-ned-train-subset.pkl $(WIKIPEDIA_PATH)/%-ned-test-subset.pkl:	$(WIKIPEDIA_PATH)/%-ned.sqlite
	ned-train-test-split --fraction-train=0.5 $^ $(WIKIPEDIA_PATH)/$*-ned-train-subset.pkl $(WIKIPEDIA_PATH)/$*-ned-test-subset.pkl

ned-train-test-split:	$(WIKIPEDIA_PATH)/de-ned-train-subset.pkl $(WIKIPEDIA_PATH)/fr-ned-train-subset.pkl $(WIKIPEDIA_PATH)/en-ned-train-subset.pkl

# ==============================================================================================================================================================

ned-pairing-train:
	ned-pairing --subset-file $(WIKIPEDIA_PATH)/de-ned-train-subset.pkl --nsamples=3000000 ned-train.sqlite $(NED_FILE) $(ENTITIES_FILE) fasttext $(N_TREES) $(DIST) $(ENTITY_INDEX_PATH)

ned-pairing-examples:
	ned-pairing-examples --nsamples=20000 ned-train.sqlite data/digisam/BERT_de_finetuned > ned-pairing-examples.txt

ned-train-test:
	ned-bert --learning-rate=3e-5 --seed=42 --train-batch-size=128 --train-size=100000 --num-train-epochs=1 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/digisam/BERT_de_finetuned ./ned-model-test --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE)

# ===============================================================================================================================================================

de-ned-train-0:
	ned-bert --learning-rate=3e-5 --seed=42 --train-batch-size=128 --train-size=100000 --num-train-epochs=400 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/digisam/BERT_de_finetuned ./ned-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKIPEDIA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl

de-ned-train-1:
	ned-bert --learning-rate=5e-6 --seed=23 --train-batch-size=128 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/BERT/NED/de-model-0 data/BERT/NED/de-model-1 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKIPEDIA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl

# ===============================================================================================================================================================

en-ned-train-0:
	ned-bert --learning-rate=1e-5 --seed=42 --train-batch-size=128 --gradient-accumulation-steps=4 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(WIKIPEDIA_PATH)/en-ned.sqlite --train-set-file $(WIKIPEDIA_PATH)/en-ned-train-subset.pkl --dev-set-file $(WIKIPEDIA_PATH)/en-ned-test-subset.pkl --test-set-file $(WIKIPEDIA_PATH)/en-ned-test-subset.pkl data/BERT/multi_cased_L-12_H-768_A-12 data/BERT/NED/en-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKIPEDIA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl --embedding-type=fasttext --embedding-model=data/fasttext/cc.en.300.bin

en-ned-train-1:
	ned-bert --learning-rate=1e-5 --seed=42 --train-batch-size=128 --gradient-accumulation-steps=4 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(WIKIPEDIA_PATH)/en-ned.sqlite --train-set-file $(WIKIPEDIA_PATH)/en-ned-train-subset.pkl --dev-set-file $(WIKIPEDIA_PATH)/en-ned-test-subset.pkl --test-set-file $(WIKIPEDIA_PATH)/en-ned-test-subset.pkl data/BERT/multi_cased_L-12_H-768_A-12 data/BERT/NED/en-model-1 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKIDATA_PATH)/en-wikipedia-ner-entities.pkl --embedding-type=fasttext --embedding-model=data/fasttext/cc.en.300.bin

# ===============================================================================================================================================================

fr-ned-train-0:
	ned-bert --learning-rate=1e-5 --seed=42 --train-batch-size=128 --gradient-accumulation-steps=4 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(WIKIPEDIA_PATH)/fr-ned.sqlite --train-set-file $(WIKIPEDIA_PATH)/fr-ned-train-subset.pkl --dev-set-file $(WIKIPEDIA_PATH)/fr-ned-test-subset.pkl --test-set-file $(WIKIPEDIA_PATH)/fr-ned-test-subset.pkl data/BERT/multi_cased_L-12_H-768_A-12 data/BERT/NED/fr-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKIPEDIA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl --embedding-type=fasttext --embedding-model=data/fasttext/cc.fr.300.bin

# ================================================================================================================================================

ned-test:
	ned-bert --seed=29 --eval-batch-size=128 --dev-size=100000 --num-train-epochs=10 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/BERT/NED/model-0 data/BERT/NED/model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE) 

ned-test-test:
	ned-bert --seed=29 --eval-batch-size=128 --dev-size=100000 --num-train-epochs=10 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) ./ned.model-test ./ned-model-test --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE)

