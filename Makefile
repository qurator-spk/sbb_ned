
DATA_PATH ?=data
WIKI_DATA_PATH ?=data/wikipedia
FASTTEXT_PATH ?= data/fasttext
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

de-fasttext-files:
	wget -nc --directory-prefix=$(FASTTEXT_PATH) https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz
	gunzip $(FASTTEXT_PATH)/cc.de.300.bin.gz
fr-fasttexti-files:
	wget -nc --directory-prefix=$(FASTTEXT_PATH) https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz
	gunzip $(FASTTEXT_PATH)/cc.fr.300.bin.gz
en-fasttexti-files:
	wget -nc --directory-prefix=$(FASTTEXT_PATH) https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
	gunzip $(FASTTEXT_PATH)/cc.en.300.bin.gz

de-fasttext-ORG:
	build-index $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.de.300.bin --split-parts 
de-fasttext-LOC:
	build-index $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.de.300.bin --split-parts
de-fasttext-PER:
	build-index $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.de.300.bin --split-parts
de-fasttext:	de-fasttext-ORG de-fasttext-LOC de-fasttext-PER

fr-fasttext-ORG:
	build-index $(WIKI_DATA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.fr.300.bin --split-parts 
fr-fasttext-LOC:
	build-index $(WIKI_DATA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.fr.300.bin --split-parts
fr-fasttext-PER:
	build-index $(WIKI_DATA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.fr.300.bin --split-parts
fr-fasttext:	fr-fasttext-ORG fr-fasttext-LOC fr-fasttext-PER

en-fasttext-ORG:
	build-index $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl fasttext ORG $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.en.300.bin --split-parts 
en-fasttext-LOC:
	build-index $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl fasttext LOC $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.en.300.bin --split-parts
en-fasttext-PER:
	build-index $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl fasttext PER $(N_TREES) $(OUTPUT_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/fasttext/cc.en.300.bin --split-parts
en-fasttext:	en-fasttext-ORG en-fasttext-LOC en-fasttext-PER

fasttext:	de-fasttext fr-fasttext en-fasttext

# ==================================================================================================================================================================

de-bert-ORG:
	build-index $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl bert ORG $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/de-model-1 --scalar-mix --split-parts --pooling=mean
de-bert-LOC:
	build-index $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl bert LOC $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/de-model-1 --scalar-mix --split-parts --pooling=mean
de-bert-PER:
	build-index $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl bert PER $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/de-model-1 --scalar-mix --split-parts --pooling=mean


fr-bert-ORG:
	build-index $(WIKI_DATA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl bert ORG $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/fr-model-0 --scalar-mix --split-parts --pooling=mean
fr-bert-LOC:
	build-index $(WIKI_DATA_PATH)/frde-wikipedia-ner-entities-no-redirects.pkl bert LOC $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/fr-model-0 --scalar-mix --split-parts --pooling=mean
fr-bert-PER:
	build-index $(WIKI_DATA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl bert PER $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/fr-model-0 --scalar-mix --split-parts --pooling=mean


en-bert-ORG:
	build-index $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl bert ORG $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/en-model-0 --scalar-mix --split-parts --pooling=mean
en-bert-LOC:
	build-index $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl bert LOC $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/en-model-0 --scalar-mix --split-parts --pooling=mean
en-bert-PER:
	build-index $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl bert PER $(N_TREES) $(ENTITY_INDEX_PATH) --n-processes=$(PROCESSES) --distance-measure=$(DIST) --model-path=data/BERT/NED/en-model-0 --scalar-mix --split-parts --pooling=mean

# ==================================================================================================================================================================

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

# =============================================================================================================================================================

$(WIKI_DATA_PATH)/ned.sqlite:
	ned-sentence-data --processes=$(PROCESSES) $(WIKI_DATA_PATH)/wikipedia-tagged.sqlite $@
$(WIKI_DATA_PATH)/de-ned.sqlite:
	ned-sentence-data --processes=$(PROCESSES) $(WIKI_DATA_PATH)/de-wikipedia-tagged.sqlite $@
$(WIKI_DATA_PATH)/fr-ned.sqlite:
	ned-sentence-data --processes=$(PROCESSES) $(WIKI_DATA_PATH)/fr-wikipedia-tagged.sqlite $@
$(WIKI_DATA_PATH)/en-ned.sqlite:
	ned-sentence-data --processes=$(PROCESSES) $(WIKI_DATA_PATH)/en-wikipedia-tagged.sqlite $@

ned-database:	$(WIKI_DATA_PATH)/de-ned.sqlite $(WIKI_DATA_PATH)/fr-ned.sqlite $(WIKI_DATA_PATH)/en-ned.sqlite

$(WIKI_DATA_PATH)/ned-train-subset.pkl $(WIKI_DATA_PATH)/ned-test-subset.pkl:	$(WIKI_DATA_PATH)/ned.sqlite
	ned-train-test-split --fraction-train=0.5 $^ $(WIKI_DATA_PATH)/ned-train-subset.pkl $(WIKI_DATA_PATH)/ned-test-subset.pkl

$(WIKI_DATA_PATH)/de-ned-train-subset.pkl $(WIKI_DATA_PATH)/de-ned-test-subset.pkl:	$(WIKI_DATA_PATH)/de-ned.sqlite
	ned-train-test-split --fraction-train=0.5 $^ $(WIKI_DATA_PATH)/de-ned-train-subset.pkl $(WIKI_DATA_PATH)/de-ned-test-subset.pkl

$(WIKI_DATA_PATH)/fr-ned-train-subset.pkl $(WIKI_DATA_PATH)/fr-ned-test-subset.pkl:	$(WIKI_DATA_PATH)/fr-ned.sqlite
	ned-train-test-split --fraction-train=0.5 $^ $(WIKI_DATA_PATH)/fr-ned-train-subset.pkl $(WIKI_DATA_PATH)/fr-ned-test-subset.pkl

$(WIKI_DATA_PATH)/en-ned-train-subset.pkl $(WIKI_DATA_PATH)/en-ned-test-subset.pkl:	$(WIKI_DATA_PATH)/en-ned.sqlite
	ned-train-test-split --fraction-train=0.5 $^ $(WIKI_DATA_PATH)/en-ned-train-subset.pkl $(WIKI_DATA_PATH)/en-ned-test-subset.pkl

ned-train-test-split:	$(WIKI_DATA_PATH)/de-ned-train-subset.pkl $(WIKI_DATA_PATH)/fr-ned-train-subset.pkl $(WIKI_DATA_PATH)/en-ned-train-subset.pkl

# ==============================================================================================================================================================

ned-pairing-train:
	ned-pairing --subset-file ned-train-subset.pkl --nsamples=3000000 ned-train.sqlite $(NED_FILE) $(ENTITIES_FILE) fasttext $(N_TREES) $(DIST) $(ENTITY_INDEX_PATH)

ned-pairing-examples:
	ned-pairing-examples --nsamples=20000 ned-train.sqlite data/digisam/BERT_de_finetuned > ned-pairing-examples.txt

ned-train-test:
	ned-bert --learning-rate=3e-5 --seed=42 --train-batch-size=128 --train-size=100000 --num-train-epochs=1 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/digisam/BERT_de_finetuned ./ned-model-test --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE)

# ===============================================================================================================================================================

de-ned-train-0:
	ned-bert --learning-rate=3e-5 --seed=42 --train-batch-size=128 --train-size=100000 --num-train-epochs=400 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/digisam/BERT_de_finetuned ./ned-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl

de-ned-train-1:
	ned-bert --learning-rate=5e-6 --seed=23 --train-batch-size=128 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/BERT/NED/de-model-0 data/BERT/NED/de-model-1 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKI_DATA_PATH)/de-wikipedia-ner-entities-no-redirects.pkl

# ===============================================================================================================================================================

en-ned-train-0:
	ned-bert --learning-rate=1e-5 --seed=42 --train-batch-size=128 --gradient-accumulation-steps=4 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(WIKI_DATA_PATH)/en-ned.sqlite --train-set-file $(WIKI_DATA_PATH)/en-ned-train-subset.pkl --dev-set-file $(WIKI_DATA_PATH)/en-ned-test-subset.pkl --test-set-file $(WIKI_DATA_PATH)/en-ned-test-subset.pkl data/BERT/multi_cased_L-12_H-768_A-12 data/BERT/NED/en-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKI_DATA_PATH)/en-wikipedia-ner-entities-no-redirects.pkl --embedding-type=fasttext --embedding-model=data/fasttext/cc.en.300.bin

# ===============================================================================================================================================================

fr-ned-train-0:
	ned-bert --learning-rate=1e-5 --seed=42 --train-batch-size=128 --gradient-accumulation-steps=4 --train-size=100000 --num-train-epochs=1000 --ned-sql-file $(WIKI_DATA_PATH)/fr-ned.sqlite --train-set-file $(WIKI_DATA_PATH)/fr-ned-train-subset.pkl --dev-set-file $(WIKI_DATA_PATH)/fr-ned-test-subset.pkl --test-set-file $(WIKI_DATA_PATH)/fr-ned-test-subset.pkl data/BERT/multi_cased_L-12_H-768_A-12 data/BERT/NED/fr-model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(WIKI_DATA_PATH)/fr-wikipedia-ner-entities-no-redirects.pkl --embedding-type=fasttext --embedding-model=data/fasttext/cc.fr.300.bin

# ===============================================================================================================================================================

CLEF_PATH ?=~/qurator/CLEF-HIPE-2020
CLEF_TARGET_PATH ?=$(DATA_PATH)/clef2020/max-pairs_150-dist_0.1

CLEF2020-tsc:
	-clef2tsv $(CLEF_PATH)/data/training-v1.0/de/HIPE-data-v1.0-dev-de.tsv $(CLEF_TARGET_PATH)/neat-HIPE-data-v1.0-dev-de.tsv
	-clef2tsv $(CLEF_PATH)/data/training-v1.0/de/HIPE-data-v1.0-train-de.tsv $(CLEF_TARGET_PATH)/neat-HIPE-data-v1.0-train-de.tsv

CLEF2020-entities:
	find-entities $(CLEF_TARGET_PATH)/neat-HIPE-data-v1.0-dev-de.tsv $(CLEF_TARGET_PATH)/ned-result-HIPE-data-v1.0-dev-de.tsv --ned-json-file=$(CLEF_TARGET_PATH)/ned-full-data-HIPE-data-v1.0-dev-de.json --ner-rest-endpoint=http://localhost/sbb-tools/ner/ner --ned-rest-endpoint=http://localhost/sbb-tools/ned
	find-entities $(CLEF_TARGET_PATH)/neat-HIPE-data-v1.0-train-de.tsv $(CLEF_TARGET_PATH)/ned-result-HIPE-data-v1.0-train-de.tsv --ned-json-file=$(CLEF_TARGET_PATH)/ned-full-data-HIPE-data-v1.0-train-de.json --ner-rest-endpoint=http://localhost/sbb-tools/ner/ner --ned-rest-endpoint=http://localhost/sbb-tools/ned

CLEF2020-train:
	sentence-stat $(CLEF_TARGET_PATH)/ned-result-HIPE-data-v1.0-train-de.tsv $(CLEF_TARGET_PATH)/ned-full-data-HIPE-data-v1.0-train-de.json $(CLEF_TARGET_PATH)/neat-HIPE-data-v1.0-train-de.tsv $(CLEF_TARGET_PATH)/decider-train-dataset.pkl

CLEF2020-dev:
	sentence-stat $(CLEF_TARGET_PATH)/ned-result-HIPE-data-v1.0-dev-de.tsv $(CLEF_TARGET_PATH)/ned-full-data-HIPE-data-v1.0-dev-de.json $(CLEF_TARGET_PATH)/neat-HIPE-data-v1.0-dev-de.tsv $(CLEF_TARGET_PATH)/decider-dev-dataset.pkl

CLEF2020-decider:
	train-decider $(CLEF_TARGET_PATH)/decider-train-dataset.pkl $(CLEF_TARGET_PATH)/decider.pkl

normalization:
	pdftotext -raw $(DATA_PATH)/char_normalization/Special-Characters-in-Aletheia.pdf $(DATA_PATH)/char_normalization/special.txt
	extract-normalization-table $(DATA_PATH)/char_normalization/special.txt $(DATA_PATH)/char_normalization/normalization-table.pkl
	adapt-normalization-table $(WIKI_DATA_PATH)/de-ned.sqlite $(DATA_PATH)/char_normalization/normalization-table.pkl $(DATA_PATH)/char_normalization/de-normalization-table.pkl

ned-test:
	ned-bert --seed=29 --eval-batch-size=128 --dev-size=100000 --num-train-epochs=10 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) data/BERT/NED/model-0 data/BERT/NED/model-0 --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE) 

ned-test-test:
	ned-bert --seed=29 --eval-batch-size=128 --dev-size=100000 --num-train-epochs=10 --ned-sql-file $(NED_FILE) --train-set-file $(NED_TRAIN_SUBSET_FILE) --dev-set-file $(NED_TEST_SUBSET_FILE) --test-set-file $(NED_TEST_SUBSET_FILE) ./ned.model-test ./ned-model-test --model-file pytorch_model.bin --entity-index-path $(ENTITY_INDEX_PATH) --entities-file $(ENTITIES_FILE)

