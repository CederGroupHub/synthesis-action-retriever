# Synthesis Action Retrieval

 * classifies words tokens into following action categories:

    *Starting*, *Mixing*, *Heating*, *Cooling*, *Purification*, *Shaping*, *Reaction*, *Miscellaneous*, *NotOperation*

 * extracts firing temperatures, times, and environment through dependency tree parsing
 
### Installation:
```
git clone https://github.com/CederGroupHub/synthesis-action-retrieval-public.git
cd synthesis-action-retrieval
python setup.py install
```

### Initilization:
```
from synthesis_action_retrieval import SynthActionRetriever

w2v_model = 'path-to-folder/models/your_word2vec_model'
classifier_model = 'path-to-folder/models/your_trained_classification_model'
spacy_model = 'path-to-folder/models/your_spacy_model'

SAR = SynthActionExtractor(w2v_model, classifier_model, spacy_model)
```

### Functions:

_synthesis_action_retriever.py_

 * **get_action_labels(sentence_tokens)**:

         finds actions tokens and classifies them

         :param sentence: list of sentence tokens
         :returns: list of actionss tuples (token_id, actionn_type) found in the sentence

_conditions_extraction.py_

 * **get_times_toks(sentence_tokens)**:
 
        finds tokens corresponding to time values
        
        :param sentence tokens: list of sentence tokens
        :returns: token_id, value, unit of time
       
 * **get_temperatures_toks(sentence_tokens)**:
 
        finds tokens corresponding to temperature values
        
        :param sentence tokens: list of sentence tokens
        :returns: token_id, value, unit of temperature
        
 * **get_environment_toks(sentence_tokens, materials)**:
 
        finds tokens corresponding to environment values
        
        :param sentence tokens: list of sentence tokens
        :param materials: list of materials in sentence
        :returns: token_id, value, unit of environment 
        
_build_graph.py_

 * **build_graph(sentence_tokens, action_tags, materials)**:
 
        builds synthesis workflow provided sentence tokens, action tags and materials list (optionally)
        :param sentence_tokens: list of strings
        :param action_tags: list of strings of same length as sentence_tokens
        :param materials: (optional) list of {"text": material, "tok_ids": list of tok ids in sentence}
        :return: list of dict        

### Example:
```
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
sent_toks = [tok for sent in Paragraph(text).raw_tokens for tok in sent]

graph = []
for raw_sent_toks, mer_sent in zip(sent_toks, mer_results):
    spacy_tokens = make_spacy_tokens(raw_sent_toks)
    actions = sar.get_action_labels(spacy_tokens)
    if mer_sent["all_materials"]:
        materials = [{"text": mat["text"], "tok_ids": mat["token_ids"]} for mat in mer_sent["all_materials"]]
    else:
        materials = None
    graph.append(build_graph(spacy_tokens, actions, materials))

pprint(graph)
```