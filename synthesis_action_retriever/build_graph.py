from synthesis_action_retriever.conditions_extraction import (
    get_times_toks,
    get_temperatures_toks,
    get_environment,
    tok2nums
)
from synthesis_action_retriever.neighboring_attribute_extraction import (
    get_neighbouring_temp,
    get_neighbouring_time,
    get_neighbouring_keywords,
    get_temp_keyverb,
    get_time_keyverb,
    get_env_keyverb,
    get_env_keyword,
    get_env_actmatch
)
from synthesis_action_retriever.utils import make_spacy_tokens
from pprint import pprint


class GraphBuilder:
    def __init__(self,
                 verbose=False):
        self.__verbose = verbose

    def __make_subsentences(self, spacy_toks, action_tags):
        sub_sentences = []
        action_ids = [i for i, t in enumerate(spacy_toks) if action_tags[i] != "" or t.pos_ == "VERB"]
        act_id = 0
        next_act = action_ids[act_id]
        for act, next_act in zip(action_ids[:-1], action_ids[1:]):
            sub_sentences.append(spacy_toks[act:next_act])
        sub_sentences.append(spacy_toks[next_act:])

        return sub_sentences

    def __get_subjects_list(self, current_act, spacy_tokens, action_tags):
        # Assuming that action subject is located in the sentence chunk starting from previous action/verb
        # and spanning till current action
        prev_verb_i = current_act.i
        prev_verb_tok = None
        while prev_verb_i > 0 and not prev_verb_tok:
            prev_verb_i = prev_verb_i - 1
            token = spacy_tokens[prev_verb_i]
            if token.pos_ == "VERB" or action_tags[prev_verb_i] != "":
                prev_verb_tok = token

        nsubjpass_list = [
            c for c in spacy_tokens[prev_verb_i:current_act.i]
            if c.dep_ in ["nsubjpass", "nsubj"]
        ]
        return nsubjpass_list

    def __peek_neighbor_nounphrase(self, graph_window, direction):
        if graph_window['{}_sent'.format(direction)]:

            neighbor_temp, neighbor_time, graph_id = get_neighbouring_keywords(
                graph_window['{}_sent'.format(direction)],
                graph_window['current_sent'],
                graph_window['{}_graph'.format(direction)],
                graph_window['current_graph']
            )
            if neighbor_temp or neighbor_time:
                if neighbor_temp:
                    graph_window['current_graph'][graph_id]['temp_values'] = neighbor_temp
                if neighbor_time:
                    graph_window['current_graph'][graph_id]['time_values'] = neighbor_time
                return graph_window['current_graph']

        return graph_window['current_graph']

    def __remove_redundant_tags(self, joined_acts, joined_acts_ids):
        acts_w_props = [
            act for act in joined_acts if
            any([
                prop for prop in [
                    act['temp_values'], act['time_values']
                ]
            ])
        ]
        if not acts_w_props:
            true_act = self.graph_data_sent[-1]
            joined_acts_ids.remove(self.graph_data_sent.index(true_act))
            for j in joined_acts_ids:
                del self.graph_data_sent[j]
            return
        elif len(acts_w_props) == 1:
            true_act = acts_w_props[0]
            joined_acts_ids.remove(self.graph_data_sent.index(true_act))
            for j in sorted(joined_acts_ids, reverse=True):
                del self.graph_data_sent[j]
            return
        else:
            if self.__verbose:
                print(
                    "In case with consecutive string of actions, multiple tokens" \
                    "were assigned properties... all tokens were retained"
                )
            return

    def __clean_redundancy(self):
        joined_acts = []
        joined_acts_ids = []
        for i, act in enumerate(self.graph_data_sent[1:], start=1):
            prev_act = self.graph_data_sent[i - 1]
            if (
                    prev_act['act_id'] == act['act_id'] - 1 and
                    prev_act['act_type'] == act['act_type']
            ):
                if prev_act not in joined_acts:
                    joined_acts.extend([prev_act, act])
                    joined_acts_ids.extend([i - 1, i])
                else:
                    joined_acts.append(act)
                    joined_acts_ids.append(i)
                    if i == len(self.graph_data_sent) - 1:
                        self.__remove_redundant_tags(joined_acts, joined_acts_ids)
            elif joined_acts:
                self.__remove_redundant_tags(joined_acts, joined_acts_ids)

    def build_graph(
            self,
            sentence,
            action_tags,
            materials=[],
            clean_redundancies=True,
        ):
        """
        Builds synthesis workflow provided sentence tokens, action tags and materials list (optionally)
        :param sentence_tokens: list of strings
        :param action_tags: list of strings of same length as sentence_tokens
        :param materials: (optional) list of {"text": material, "tok_ids": list of tok ids in sentence}
        :param clean_redundancies: (optional) boolean option to return only relevant action token in case of word-phrase
        :return: list of dict
        """

        sentence_tokens = make_spacy_tokens(sentence)

        if len(action_tags) != len(sentence_tokens):
            print("Mismatch between amount of tokens and action tags!")
            return []

        self.graph_data_sent = []
        mixing_materials = materials

        action_seq = [(i, act) for i, act in enumerate(action_tags) if act != ""]
        if action_seq:
            spacy_tokens = make_spacy_tokens(sentence_tokens)

            # Splitting sentence into subsentences from action/verb token to action/verb token
            # Assuming that all the action attributes are mentioned in this subsentence
            # sub_sentences = self.__make_subsentences(sentence_tokens, action_seq_tok)
            sub_sentences = self.__make_subsentences(spacy_tokens, action_tags)
        else:
            return self.graph_data_sent

        # Extracting attributes for an action from its subsentence
        action_seq_tok = [
            (t, action_tags[t.i]) for t in spacy_tokens if
            action_tags[t.i] != "" or t.pos_ == "VERB"
        ]
        if len(action_seq_tok) != len(sub_sentences):
            print("Mismatch between number of action and number subsentences!")

        start_subsent = action_seq_tok[0][0].i
        prev_subjects_list = []
        prev_act_tok = ""
        for num_act, ((act_tok, act_type), sub_sent) in enumerate(zip(action_seq_tok, sub_sentences)):
            temp_toks, time_toks = [], []
            env_ids = [[], []]
            env_toks = ["", ""]
            sub_sent_text = [t.text for t in sub_sent]

            # Assuming that action subject is located in the sentence chunk starting from previous action/verb
            # and spanning till current action
            subjects_list = self.__get_subjects_list(act_tok, spacy_tokens, action_tags)

            if self.__verbose:
                print("->", act_tok, num_act, act_tok.i)
                print("\tSubsentence:", sub_sent)
                print("\tSubjects list:", subjects_list)
                print("\tPrevious subjects list:", prev_subjects_list)

            # Action subject is a subtree of nsubjpass noun
            # if action refer to previous step reference_act = True, else reference_act = True
            act_subject, reference_act = ("", False)
            if not subjects_list:
                subjects_list = prev_subjects_list
                if prev_act_tok == act_tok.text or num_act == 0:
                    reference_act = True

            # if repeating or reference action, no subject assigned
            if 're' != act_tok.text[0:2] and subjects_list and not reference_act:
                if self.__verbose:
                    print("\tNsubjpass subtree:", [t.text for tok in subjects_list for t in tok.subtree])
                act_subject = " ".join([t.text for tok in subjects_list for t in tok.subtree if t.i < act_tok.i])
            prev_subjects_list = subjects_list  # if subjects_list else prev_subjects_list
            prev_act_tok = act_tok.text

            if self.__verbose:
                print("\tAssigned Subject:", act_subject, "--->", act_tok.text)
                print("\tReference action:", reference_act)

            # Finding mixing conditions and type
            # Mixing types: solid mix, solution mix, mix with liquid
            if act_type in ["Mixing", "Heating", "Cooling"] or any(w in act_tok.text for w in ["dry", "evaporate"]):
                temp_toks = [t for t in get_temperatures_toks(sub_sent_text)]
                time_toks = [t for t in
                             get_times_toks(sub_sent_text)]
                env_ids, env_toks = get_environment(sub_sent, mixing_materials)

            if self.__verbose:
                print("\tEnvironment:", env_toks)

            # converting temperature and time tokens into numerical data structure
            temperature_values = tok2nums(temp_toks, sub_sent)
            time_values = tok2nums(time_toks, sub_sent)

            if self.__verbose:
                print("\tTemp values:", temperature_values)
                print("\tTime values:", time_values)

            if act_type != "":
                self.graph_data_sent.append(dict(
                    act_id=act_tok.i,
                    act_type=act_type,
                    subsent=[sub_sent[0].i, sub_sent[-1].i + 1],
                    subject=act_subject,
                    act_token=act_tok.text,
                    temp_values=temperature_values,  # list of {max: float, min: float, values: [float], tok_ids: [int]}
                    time_values=time_values,  # list of {max: float, min: float, values: [float], tok_ids: [int]}
                    env_toks=env_toks,  # [str "in", str "with"]
                    env_ids=env_ids,  # [token ids for "in", token ids for "with"]
                    ref_act=reference_act
                )
                )

            start_subsent = start_subsent + len(sub_sent)

        # remove redundant action tokens from final graph
        #    if sequence of consecutive actions, keep the one the has attributes or, if none,
        #    keep the final tagged token
        if clean_redundancies:
            self.__clean_redundancy()

        return self.graph_data_sent

    def refine_graph(self, full_graph, sentences_materials):
        """
        Refine synthesis workflow graph to incorporate attributes from neighboring sentences

        :param full_graph: list of dicts returned from build_graph function
        :param sentences_materials: list of dicts of sentences and associated materials
        :return: list of dict
        """
        sentences = [s["sentence"] for s in sentences_materials]
        sent_toks = []
        for sent in sentences:
            sent_toks.append([t.text for t in make_spacy_tokens(sent)])

        for graph in full_graph:
            cursor = full_graph.index(graph)
            if graph:
                if cursor - 1 >= 0:
                    prev_sent = sentences[cursor - 1]
                    prev_sent_toks = sent_toks[cursor - 1]
                    prev_graph = full_graph[cursor - 1]
                else:
                    prev_sent, prev_sent_toks, prev_graph = None, None, None

                if cursor + 1 <= len(full_graph) - 1:
                    next_sent = sentences[cursor + 1]
                    next_sent_toks = sent_toks[cursor + 1]
                    next_graph = full_graph[cursor + 1]
                else:
                    next_sent, next_sent_toks, next_graph = None, None, None

                graph_window = dict(
                    prev_sent=prev_sent,
                    prev_sent_toks=prev_sent_toks,
                    prev_graph=prev_graph,
                    current_sent=sentences[cursor],
                    current_sent_toks=sent_toks[cursor],
                    current_graph=graph,
                    next_sent=next_sent,
                    next_sent_toks=next_sent_toks,
                    next_graph=next_graph
                )

                if any([(node['temp_values'] == [] or node['time_values'] == 0) for node in graph]):
                    if graph != self.__peek_neighbor_nounphrase(graph_window, 'prev'):
                        graph = self.__peek_neighbor_nounphrase(graph_window, 'prev')
                    elif graph != self.__peek_neighbor_nounphrase(graph_window, 'next'):
                        graph = self.__peek_neighbor_nounphrase(graph_window, 'next')
                    elif graph_window['next_sent']:
                        if any([node['temp_values'] == [] for node in graph]):
                            act_search_temp, graph = get_neighbouring_temp(
                                graph_window['next_sent_toks'],
                                graph_window['next_graph'],
                                graph
                            )
                            verb_search_temp, graph = get_temp_keyverb(
                                graph_window['next_sent_toks'],
                                graph_window['next_graph'],
                                graph
                            )
                        if any([node['time_values'] == [] for node in graph]):
                            act_search_time, graph = get_neighbouring_time(
                                graph_window['next_sent_toks'],
                                graph_window['next_graph'],
                                graph
                            )
                            verb_search_time, graph = get_time_keyverb(
                                graph_window['next_sent_toks'],
                                graph_window['next_graph'],
                                graph
                            )
                if any([node['env_toks'] == ['', ''] for node in graph]):
                    if graph_window['next_sent']:
                        next_mats = [
                            s["materials"] for s in sentences_materials if
                            s["sentence"] == graph_window['next_sent']
                        ][0]
                        act_match_env_ids, act_match_env_toks, graph = get_env_actmatch(
                            graph_window['next_sent'],
                            graph_window['next_graph'],
                            graph,
                            next_mats
                        )
                        if act_match_env_toks:
                            break
                        keyverb_ids, keyverb_env_toks, graph = get_env_keyverb(
                            graph_window['next_sent'],
                            graph_window['next_graph'],
                            graph,
                            next_mats
                        )
                        if keyverb_env_toks:
                            break
                        keyword_ids, keyword_env_toks, graph = get_env_keyword(
                            graph_window['next_sent_toks'],
                            graph_window['next_graph'],
                            graph,
                            next_mats
                        )
        return full_graph