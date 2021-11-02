import json
from pprint import pprint
from chemdataextractor.doc import Paragraph
from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.build_graph import GraphBuilder
from synthesis_action_retriever.utils import make_spacy_tokens

sar = SynthActionRetriever(
    embedding_model='path-to-model',
    extractor_model='path-to-model'
)
gb = GraphBuilder()

with open('./data/example_sentences.json', 'r') as fp:
    examples = json.load(fp)

graph = []
for sent in examples:
    # Note: one can feed in a whole paragraph to line below and receive a list of tokenized sentences
    sent_toks = Paragraph(sent["sentence"]).raw_tokens[0]
    spacy_tokens = make_spacy_tokens(sent_toks)
    actions = sar.get_action_labels(spacy_tokens)
    graph.append(gb.build_graph(spacy_tokens, actions, sent["materials"]))

para = ' '.join([s["sentence"] for s in examples])
para_sent_toks = Paragraph(para).raw_tokens

refined_graph = gb.refine_graph(graph, examples, para_sent_toks)

pprint(refined_graph)
