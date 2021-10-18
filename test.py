from text_cleanup import TextCleanUp
from pprint import pprint
from chemdataextractor.doc import Paragraph

from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.build_graph import GraphBuilder
from synthesis_action_retriever.utils import make_spacy_tokens

import json

tp = TextCleanUp()
sar = SynthActionRetriever()
gb = GraphBuilder()

with open('./data/example_paragraph.json', 'r') as fp:
    sample_text = json.load(fp)

text = tp.cleanup_text(sample_text['sentence'])
sent_toks = [tok for sent in Paragraph(text).raw_tokens for tok in sent]

spacy_tokens = make_spacy_tokens(sent_toks)
actions = sar.get_action_labels(spacy_tokens)

pprint(actions)
