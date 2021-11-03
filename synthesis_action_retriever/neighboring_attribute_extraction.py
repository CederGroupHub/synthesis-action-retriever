import spacy
from nltk.stem.snowball import SnowballStemmer

from synthesis_action_retriever.synthesis_action_retriever import SynthActionRetriever
from synthesis_action_retriever.conditions_extraction import get_times_toks, get_temperatures_toks, get_environment, tok2nums

from pprint import pprint

stemmer = SnowballStemmer(language='english')
nlp = spacy.load("en_core_web_sm")

def get_neighbouring_temp(sentences, next_graph, main_graph):
    # sentences: sentence to the right, should be input as a list of tokens
    # next_graph: synthesis graph of the input sentence. Should be empty for this function to run
    # main_graph: synthesis graph of sentence with the action token
    temp = []
    new_graph = main_graph
    if not next_graph:  # ensures temps are not taken from the graph of a different action
        for index, data in enumerate(main_graph):
            for token in sentences:
                if stemmer.stem(token) == stemmer.stem(data['act_token']):
                    temp_toks = get_temperatures_toks(sentences)
                    temp = tok2nums(temp_toks, spacy.tokens.Doc(nlp.vocab, sentences))
                    new_graph[index]['temp_values'].extend(temp)

    return temp, new_graph


def get_neighbouring_time(sentences, next_graph, main_graph):
    time = []
    new_graph = main_graph
    if not next_graph:  # ensures times are not taken from the graph of a different action
        for index, data in enumerate(main_graph):
            for token in sentences:
                if stemmer.stem(token) == stemmer.stem(data['act_token']):
                    time_toks = get_times_toks(sentences)
                    time = tok2nums(time_toks, spacy.tokens.Doc(nlp.vocab, sentences))
                    new_graph[index]['time_values'].extend(time)

    return time, new_graph

def get_neighbouring_keywords(neighbouring_sentence,sentence,neighbouring_graph,graph):
    #neighbouring_sentence,sentence, are strings from ext_paragraph
    #graph, neighbouring_graph are the synthesis_graphs of the main and neighbouring sentences

    #tokenize the input sentences
    sentence_tokens = []
    neighbouring_sentence_tokens = []

    for token in nlp(sentence):
        sentence_tokens.append(token.text)

    for token in nlp(neighbouring_sentence):
        neighbouring_sentence_tokens.append(token.text)
    
    graph_id = 0
    key_words = []
    temp = []
    time = []

    units = ["h", "hr", "hrs", "min", "hour", "hours", "minutes", "d", "day", "days", "°C", "C", "K", "Torr", "a", "A", "It", "it", "The", "the", "They", "they", "g", "Them", "them", "pressure"]

    if not neighbouring_graph: #ensures temp/times are not taken from the graph of a different action
        #get actions list
        #actions = oe.get_action_labels(sentence_tokens)
        
        #make subsentences in main graph

        spacy_tokens = spacy.tokens.Doc(nlp.vocab,sentence_tokens)

        subsent = []
        actions_ids = []
        for data in graph:
            actions_ids.append(data['act_id'])

        act_id = 0
        next_act = actions_ids[act_id]
        for act, next_act in zip(actions_ids[:-1],actions_ids[1:]):
            subsent.append(spacy_tokens[act:next_act])
        subsent.append(spacy_tokens[next_act:])
    
        #get keyword from neighbouring sentence
        for chunk in nlp(neighbouring_sentence).noun_chunks:
            if chunk.root.text.isalpha() and chunk.root.text not in units:
                key_words.extend([chunk.root.text])

        #check main sentence for keyword matches
        for num,sub_sentence in enumerate(subsent):
            for chunk in nlp(sentence).noun_chunks:
                for key in key_words:
                    if key in chunk.text:
                        graph_id = num
                        temp_toks = get_temperatures_toks(neighbouring_sentence_tokens)
                        temp = tok2nums(temp_toks,spacy.tokens.Doc(nlp.vocab, neighbouring_sentence_tokens))
                        time_toks = get_times_toks(neighbouring_sentence_tokens)
                        time = tok2nums(time_toks,spacy.tokens.Doc(nlp.vocab, neighbouring_sentence_tokens))

    return temp, time, graph_id


def get_temp_keyverb(next_sentence, next_graph, main_graph):
    # neighbouring_sentence taken from ext_paragraph. Should be input as strings
    # main_graph, next_graph are the synthesis_graphs of the main and neighbouring sentences
    temp_keywords = ['set', 'maintained', 'fixed', 'carried out', 'occurred', 'took place', 'conducted', 'done']
    temp = []
    new_graph = main_graph

    if not next_graph:  # ensures temps are not taken from the graph of a different action
        if 'reaction' in next_sentence or 'step' in next_sentence:
            for graph_id in range(len(main_graph)):
                if main_graph[graph_id]['act_type'] in ['Heating', 'Purification', 'Reaction']:
                    for num, token in enumerate(next_sentence):
                        if token in temp_keywords:
                            temp_toks = [t for t in get_temperatures_toks(next_sentence)]
                            temp = tok2nums(temp_toks, spacy.tokens.Doc(nlp.vocab, next_sentence))
                            new_graph[graph_id]['temp_values'].extend(temp)

    return temp, new_graph


def get_time_keyverb(next_sentence, next_graph, main_graph):
    time_keywords = ['maintained', 'fixed', 'carried out', 'occurred', 'took place', 'conducted', 'done']
    time = []
    new_graph = main_graph

    if not next_graph:  # ensures times are not taken from the graph of a different action
        if 'reaction' in next_sentence or 'step' in next_sentence:
            for graph_id in range(len(main_graph)):
                if main_graph[graph_id]['act_type'] in ['Heating', 'Purification', 'Reaction']:
                    for num, token in enumerate(next_sentence):
                        if token in time_keywords:
                            time_toks = get_times_toks(next_sentence)
                            time = tok2nums(time_toks, spacy.tokens.Doc(nlp.vocab, next_sentence))
                            new_graph[graph_id]['time_values'].extend(time)

    return time, new_graph


def get_env_keyverb(next_sentence, next_graph, main_graph, materials):
    # next_sentence is taken from ext_paragraph. Should be input as string
    # main_graph, next_graph are the synthesis_graphs of the main and neighbouring sentences
    # materials is materials list from MER

    env_keywords = ['occurred', 'conducted', 'done']
    env_ids, env_toks = [], []
    sent_toks = [token.text for token in nlp(next_sentence)]
    new_graph = main_graph

    if not next_graph:  # ensures envs are not taken from the graph of a different action
        for index, data in enumerate(main_graph):
            if data['act_type'] in ['Heating', 'Mixing', 'Reaction']:
                if 'reaction' in next_sentence or 'step' in next_sentence:
                    for token in nlp(next_sentence):
                        if token.text in env_keywords:
                            env_ids, env_toks = get_environment(nlp(next_sentence), materials)

                    if not env_ids:
                        for num, token in enumerate(sent_toks):
                            if token == 'carried':
                                if sent_toks[num + 1] == 'out':
                                    env_ids, env_toks = get_environment(nlp(next_sentence), materials)

                    if not env_ids:
                        for num, token in enumerate(sent_toks):
                            if token == 'took':
                                if sent_toks[num + 1] == 'place':
                                    env_ids, env_toks = get_environment(nlp(next_sentence), materials)

                new_graph[index]['env_toks'] = env_toks
                new_graph[index]['env_ids'] = env_ids

    return env_ids, env_toks, new_graph


def get_env_actmatch(next_sentence, next_graph, main_graph, materials):
    # next_sentence: sentence to the right, should be input as a string
    # next_graph: synthesis graph of the input sentence. Should be empty for this function to run
    # main_graph: synthesis graph of sentence with the action token
    # materials: materials tokens list from MER

    env_ids, env_toks = [], []
    new_graph = main_graph
    if not next_graph:  # ensures envs are not taken from the graph of a different action
        for index, data in enumerate(main_graph):
            for token in nlp(next_sentence):
                if stemmer.stem(token.text) == stemmer.stem(data['act_token']):
                    env_ids, env_toks = get_environment(nlp(next_sentence), materials)
                    new_graph[index]['env_toks'] = env_toks
                    new_graph[index]['env_ids'] = env_ids

    return env_ids, env_toks, new_graph


def get_env_keyword(next_sentence, next_graph, main_graph, materials):
    # next_sentence is a list of tokens
    # main_graph, next_graph are the synthesis_graphs of the main and neighbouring sentences
    # materials is materials list from MER
    mat_ids, mat_toks = [], []
    env_ids, env_toks = [], []
    new_graph = main_graph
    index = 0

    for m in materials:
        mat_ids.extend(m['tok_ids'])
        mat_toks.append(m['text'])

    env_keywords = ['agent', 'solvent', 'medium', 'flux', 'buffer', 'fuel']

    # not_env = ['dopant','catalyst','precursor','raw','starting']

    def split_sentence(sentence_tokens, mat_toks):
        # split action sentence into subsentences
        sub_sentences = []
        mat_ids = [i for i, tok in enumerate(sentence_tokens) if tok in mat_toks]
        if not mat_ids:
            return sub_sentences, mat_ids
        else:
            mat_id = 0
            next_mat = mat_ids[mat_id]
            for mat, next_mat in zip(mat_ids[:-1], mat_ids[1:]):
                sub_sentences.append(sentence_tokens[mat:next_mat])
            sub_sentences.append(sentence_tokens[next_mat:])

            return sub_sentences, mat_ids

    if not next_graph:  # ensures envs are not taken from the graph of a different action
        sub_sentences, mat_ids = split_sentence(next_sentence, mat_toks)
        for i, sent in enumerate(sub_sentences):
            if main_graph[i]['act_type'] in ['Mixing', 'Heating', 'Reaction']:
                for num, token in enumerate(sent):
                    if token in env_keywords:
                        for tok in sent[:num]:
                            if tok in mat_toks:
                                env_toks.append(tok)
                                env_ids.append(mat_ids[i])
                                index = i
                                new_graph[index]['env_toks'] = env_toks
                                new_graph[index]['env_ids'] = env_ids


    return env_ids, env_toks, new_graph