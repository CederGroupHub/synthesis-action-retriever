# Synthesis Action Retrieval

 * classifies words tokens into following action categories:

    *Starting*, *Mixing*, *Heating*, *Cooling*, *Purification*, *Shaping*, *Reaction*, *Miscellaneous*, *NotOperation*

 * extracts firing temperatures, times, and environment through dependency tree parsing
 
### Installation:
```
git clone https://github.com/CederGroupHub/synthesis-action-retriever.git
cd synthesis-action-retriever
python setup.py install
```

### Initilization:
```
from synthesis_action_retriever import SynthActionRetriever

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
```

### Citation:

If you find the codes and data useful, please cite our paper:

```
@Article{D1DD00034A,
author ="Wang, Zheren and Cruse, Kevin and Fei, Yuxing and Chia, Ann and Zeng, Yan and Huo, Haoyan and He, Tanjin and Deng, Bowen and Kononova, Olga and Ceder, Gerbrand",
title  ="ULSA: unified language of synthesis actions for the representation of inorganic synthesis protocols",
journal  ="Digital Discovery",
year  ="2022",
pages  ="-",
publisher  ="RSC",
doi  ="10.1039/D1DD00034A",
```
