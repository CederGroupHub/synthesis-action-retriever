import os
import json
from pprint import pprint
from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.build_graph import GraphBuilder
from synthesis_action_retriever.utils import make_spacy_tokens

dir_path = "path-to-models"
w2v_model = "path-to-w2v_model"
ext_model = "path-to-ext_model"

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

refined_graph = gb.refine_graph(graph, examples)
pprint(refined_graph)
