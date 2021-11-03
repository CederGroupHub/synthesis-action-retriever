import os
import json
from pprint import pprint
from chemdataextractor.doc import Paragraph
from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.build_graph import GraphBuilder
from synthesis_action_retriever.utils import make_spacy_tokens

dir_path = "path-to-model"
w2v_model = "w2v_model_name"
ext_model = "ext_model_name"

sar = SynthActionRetriever(
    embedding_model=os.path.join(dir_path, w2v_model),
    extractor_model=os.path.join(dir_path, ext_model)
)
gb = GraphBuilder()

with open('./data/example_sentences.json', 'r') as fp:
    examples = json.load(fp)

graph = []
for sent in examples:
    spacy_tokens = make_spacy_tokens(sent["sentence"])
    actions = sar.get_action_labels(spacy_tokens)
    graph.append(gb.build_graph(spacy_tokens, actions, sent["materials"]))

para = ' '.join([s["sentence"] for s in examples])
para_sent_toks = Paragraph(para).raw_tokens

refined_graph = gb.refine_graph(graph, examples, para_sent_toks)

pprint(refined_graph)
