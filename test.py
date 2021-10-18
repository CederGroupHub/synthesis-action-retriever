from text_cleanup import TextCleanUp
import json
from pprint import pprint
from chemdataextractor.doc import Paragraph
from materials_entity_recognition import MatRecognition
from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.build_graph import GraphBuilder
from synthesis_action_retriever.utils import make_spacy_tokens

tp = TextCleanUp()
mer = MatRecognition()
sar = SynthActionRetriever()
gb = GraphBuilder()

with open('./data/example_paragraph.txt', 'r') as fp:
    sample_text = fp.read()

mer_results = mer.mat_recognize(sample_text)

text = tp.cleanup_text(sample_text)
sent_toks = Paragraph(text).raw_tokens
graph = []
for raw_sent_toks, mer_sent in zip(sent_toks, mer_results):
    spacy_tokens = make_spacy_tokens(raw_sent_toks)
    actions = sar.get_action_labels(spacy_tokens)
    if mer_sent["all_materials"]:
        materials = [{"text": mat["text"], "tok_ids": mat["token_ids"]} for mat in mer_sent["all_materials"]]
    else:
        materials = []
    graph.append(gb.build_graph(spacy_tokens, actions, materials))

pprint(graph)
