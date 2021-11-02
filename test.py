import json
from pprint import pprint
from chemdataextractor.doc import Paragraph
from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.build_graph import GraphBuilder
from synthesis_action_retriever.utils import make_spacy_tokens

sar = SynthActionRetriever(
    embedding_model='/Users/kevcruse96/Desktop/D2S2/Saved_Models/sar_models/w2v_embeddings_v2_words_420K',
    extractor_model='/Users/kevcruse96/Desktop/D2S2/Saved_Models/sar_models/Bi-RNN_cl7_ed100_TF_20211018-122820'
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

pprint(graph)
