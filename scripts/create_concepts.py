# conda install nmslib conda-forge::cupy nvidia/label/cuda-12.8.0::cuda-toolkit
# pip install datasets scispacy==0.5.5 spacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

import tqdm
import spacy
from datasets import load_dataset
from scispacy.umls_linking import UmlsEntityLinker

ds = load_dataset("casperhansen/pmc-oa-markdown", split="train")
ds = ds.take(1000)
texts = (rec["text"] for rec in ds)

# ──────────── STEP 1: GPU NER ────────────
spacy.require_gpu()

nlp_ner = spacy.load(
    "en_core_sci_md",
    disable=[
        "tok2vec",
        "tagger",
        "parser",
        "attribute_ruler",
        "lemmatizer",
    ],
)

# batch size 512 requires 46GB VRAM
docs = list(
    tqdm.tqdm(
        nlp_ner.pipe(texts, batch_size=512, n_process=1),
        total=len(ds),
        desc="Named Entity Recognition (NER) running on GPU",
    )
)

# ──────────── STEP 2: CPU UMLS LINKING ────────────
nlp_link = spacy.blank("en")
linker = UmlsEntityLinker(
    resolve_abbreviations=True,
    max_entities_per_mention=1,
    k=10,
    threshold=0.7,
)
nlp_link.add_pipe(
    "scispacy_linker",
    config={
        "linker_name": "umls",
    },
    last=True,
)

concept_lists = list(
    tqdm.tqdm(
        nlp_link.pipe(
            docs,
            batch_size=1,
            n_process=8, 
        ),
        total=len(docs),
        desc="UMLS Linking",
    )
)

final_cuis = [
    [cui for ent in doc.ents for (cui, _) in ent._.kb_ents[:1]]
    for doc in concept_lists
]
