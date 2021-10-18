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
from pprint import pprint
from chemdataextractor.doc import Paragraph

from operations_extractor import OperationsExtractor
oe = OperationsExtractor()

tp = TextCleanUp()

text_sents = ["LiNixMn2−xO4 (x=0.05,0.1,0.3,0.5) samples were prepared in either an air or an O2 atmosphere by solid-state reactions.",
              "Mixtures of Li2CO3,MnCO3, and NiO were heated at 700°C for 24 to 48 h with intermittent grinding.",
              "All these samples were cooled to room temperature at a controlled rate of 1°C/min.",
              "Unless specifically stated, all the samples described below were prepared in an atmosphere of air."]

paragraph_data = []
for sent in text_sents:

    text = tp.cleanup_text(sent)
    sent_toks = [tok for sent in Paragraph(text).raw_tokens for tok in sent]
    operations, spacy_tokens = oe.get_operations(sent_toks)
    updated_operations = oe.operations_correction(spacy_tokens, operations, parsed_tokens=True)
    updated_operations = oe.find_aqueous_mixing(spacy_tokens, updated_operations, parsed_tokens=True)
    paragraph_data.append((spacy_tokens, updated_operations))

paragraph_data_upd = oe.operations_refinement(paragraph_data, parsed_tokens=True)

pprint(paragraph_data_upd)
```