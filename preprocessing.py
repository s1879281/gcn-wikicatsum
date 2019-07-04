# -*- coding: utf-8 -*-
from stanfordcorenlp import StanfordCoreNLP
import json
import networkx as nx

import re
import string

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess.py')
    parser.add_argument('--split', default="train", help="Split of the data to be preprocessed. Options: [train|valid|test].")
    parser.add_argument('--data-dir', default="data/film_tok_min5_L7.5k", help="Path to data directory.")
    parser.add_argument('--no-raw', action='store_true', help="Do not to preprocess the raw dataset.")
    parser.add_argument('--save-dir', default="data/film_tok_min5_L7.5k/relations", help="Path to save results.")

    args = parser.parse_args()

    return args

def _normalize_text_cleaned(text):
   # Space around punctuation
    text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"< EOP >", "<EOP>", text)
    text = re.sub(r"< EOT >", "<EOT>", text)
    text = text.strip()
    return text

def _count_lines(*files):
    """
    Read several files, assert the count of lines of each file is the same, and return the count of lines.

    :param files: Path to the files.
    :return: The count of lines of any of the files if they are the same.
    """

    counts = []
    try:
        for file in files:
            with open(file, 'rb') as f:
                counts.append(len(f.readlines()))
        assert len(set(counts)) == 1
        return counts[0]
    except FileNotFoundError:
        print("Creating new files. Start processing.")
        return 0

def readfile(path):
    """
    Read source data, split titles and documents, and save them into a list.

    :param path: Path to the file.
    :param n: Read first n instances.
    :return: A list of lists, each of which contains the title and first 800 tokens of each instance.
    """

    texts = []
    with open(path, 'rb') as f:
        for line in f.readlines():
            line = _normalize_text_cleaned(line.decode()).split('<EOT>')
            title, doc = line[0].strip(), line[1].strip()     # Split title and document
            doc = ' '.join(doc.split()[:800])                 # Restrict the size of input to the first 800 tokens
            texts.append((title, doc))
    return texts

def getConnections(args, text, text_index):
    """
    Split the text into paragraphs, extract open-domain relation triples and coreference pairs from each paragraph.
    Then extract string matching pairs from the (s,p,o) triples. Write the results into files.

    :param text:
    :param text_index: The index of the text (0-start).
    :return: Open-domain relation triples and coreference relation pairs.
            Each open-domain relation triple (s,p,o) is represented by a dictionary with entries: "subject", "subjectSpan",
            "relation", "relationSpan", "object", "objectSpan" and "paragraphIndex".
            Each coreference relation is represented by a dictionary with entries:  'representativeMention',
            'representativeMentionSpan', 'mentioned', 'mentionedSpan'.
    """

    # Save the index of the beginning token of each paragraph into a list
    paragraphs = [paragraph.strip() for paragraph in text.split('<EOP>')]
    paragraph_lengths = [len(paragraph.split()) for paragraph in paragraphs]
    paragraph_start_indices = [0] + paragraph_lengths
    paragraph_start_indices.pop()
    for i in range(1, len(paragraph_start_indices)):
        paragraph_start_indices[i] += paragraph_start_indices[i - 1]

    nlp = CoreNLPService()
    triples = []
    coref = []

    for index, paragraph in enumerate(paragraphs):
        if not paragraph or len(paragraph) > 3000:
            continue
        triples += nlp.getOpenie(paragraph, index, paragraph_start_indices[index])
        coref += nlp.getCoref(paragraph, paragraph_start_indices[index])

    coref += interParagraphStringMatch(triples)      # Append the string matches to the coreference list

    # Wrire triples and coreference into txt files
    triples_path = os.path.join(args.save_dir, args.split, 'triples.txt')
    coref_path = os.path.join(args.save_dir, args.split, 'coref.txt')

    with open(triples_path, 'a+', encoding='utf8') as f:
        strTriples = [str(triple) for triple in triples]
        f.write('---------Instance %d---------\n' % (text_index + 1))
        f.write('\n'.join(strTriples))
        f.write('\n')

    with open(coref_path, 'a+', encoding='utf8') as f:
        strCoref = [str(pair) for pair in coref]
        f.write('---------Instance %d---------\n' % (text_index + 1))
        f.write('\n'.join(strCoref))
        f.write('\n')

    return triples, coref

def buildGraphwithNE(args, text, text_index):
    """
    From a given text, extract both open-domain relation triples and coreference relation pairs, and build graphs
    from them. The graphs are stored and returned using four lists: nodes, labels, nodes1 and nodes2.

    :param text:
    :param text_index: The index of the text (0-start).
    :return: nodes: Each element of the list is a token sequence, representing tokens from a relation.
            labels: Each element of the list is a label sequence, representing edge labels of directed edges from a relation.
                     The labels include 'A0', 'A1', 'coref' and 'NE'.
            nodes1: Each element of the list is a position sequence, representing positions of the head nodes of directed edges from a relation.
            nodes2: Each element of the list is a position sequence, representing positions of the tail nodes of directed edges from a relation.
    """

    triples, coref = getConnections(args, text, text_index)

    DG = nx.MultiDiGraph()
    for triple in triples:
        DG.add_edge((triple['subject'], triple['subjectSpan'][0]),
                    (triple['object'], triple['objectSpan'][0]),
                    label=(triple['relation'], triple['relationSpan'][0]))
    for pair in coref:
        DG.add_edge((pair['representativeMention'], pair['representativeMentionSpan'][0]),
                     (pair['mentioned'], pair['mentionedSpan'][0]),
                     label=('coref', None))

    nodes = []
    labels = []
    nodes1 = []
    nodes2 = []

    for eTriple in DG.edges(data='label'):
        srcNodes = []
        srcEdgesLabels = []
        srcEdgesNode1 = []
        srcEdgesNode2 = []

        rel = [x.strip() for x in eTriple[2][0].split()]
        subj = [x.strip() for x in eTriple[0][0].split()]
        obj = [x.strip() for x in eTriple[1][0].split()]

        subjNodeDescendants = []
        objNodeDescendants = []
        relNodeDescendants = []

        subjNode = subj[0]
        if len(subj) > 1:
            subjNodeDescendants = subj[1:]
        objNode = obj[0]
        if len(obj) > 1:
            objNodeDescendants = obj[1:]
        relNode = rel[0]
        if len(rel) > 1:
            relNodeDescendants = rel[1:]

        if relNode != 'coref':
            srcNodes.append(subjNode)
            subjIdx = eTriple[0][1]
            srcNodes.append(relNode)
            relIdx = eTriple[2][1]
            srcNodes.append(objNode)
            objIdx = eTriple[1][1]

            srcEdgesLabels.append("A0")
            srcEdgesNode1.append(str(subjIdx))
            srcEdgesNode2.append(str(relIdx))

            srcEdgesLabels.append("A1")
            srcEdgesNode1.append(str(objIdx))
            srcEdgesNode2.append(str(relIdx))

        else:
            srcNodes.append(subjNode)
            subjIdx = eTriple[0][1]
            srcNodes.append(objNode)
            objIdx = eTriple[1][1]

            srcEdgesLabels.append("coref")
            srcEdgesNode1.append(str(subjIdx))
            srcEdgesNode2.append(str(objIdx))

            srcEdgesLabels.append("coref")
            srcEdgesNode1.append(str(objIdx))
            srcEdgesNode2.append(str(subjIdx))

        for neNode in subjNodeDescendants:
            srcNodes.append(neNode)
            subjIdx += 1
            srcEdgesLabels.append("NE")
            srcEdgesNode1.append(str(subjIdx))
            srcEdgesNode2.append(str(eTriple[0][1]))

        # Only for (s,p,o) triples
        for neNode in relNodeDescendants:
            srcNodes.append(neNode)
            relIdx += 1
            srcEdgesLabels.append("RL")
            srcEdgesNode1.append(str(relIdx))
            srcEdgesNode2.append(str(eTriple[2][1]))

        for neNode in objNodeDescendants:
            srcNodes.append(neNode)
            objIdx += 1
            srcEdgesLabels.append("NE")
            srcEdgesNode1.append(str(objIdx))
            srcEdgesNode2.append(str(eTriple[1][1]))

        nodes.append(" ".join(srcNodes))
        labels.append(" ".join(srcEdgesLabels))
        nodes1.append(" ".join(srcEdgesNode1))
        nodes2.append(" ".join(srcEdgesNode2))

    return nodes, labels, nodes1, nodes2


pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'that', 'this', 'one', 'those']

def interParagraphStringMatch(triples):
    """
    From the triples, find out all of the same entity pairs from different paragraphs.
    The entity of each pair that appears first is stored as 'representativeMention', while the other is stored as 'mentioned'.

    :param triples: Each relation is represented by a dictionary with entries: "subject", "subjectSpan", "relation", "relationSpan",
                "object", "objectSpan" and "paragraphIndex".
    :return: Each coreference relation is represented by a dictionary with entries:
                    'representativeMention', 'representativeMentionSpan', 'mentioned', 'mentionedSpan'.
    """
    tripleDict = {}
    stringMatches = []
    for triple in triples:
        for entity in ['subject', 'object']:
            if triple[entity] not in tripleDict and triple[entity] not in pronouns:
                tripleDict.update({triple[entity]: {'span':triple[entity + 'Span'], 'paragraphIndex':triple['paragraphIndex']}})
            elif triple[entity] in tripleDict and triple['paragraphIndex'] != tripleDict[triple[entity]]['paragraphIndex']:
                stringMatches.append({'representativeMention': triple[entity],
                                      'representativeMentionSpan': tripleDict[triple[entity]]['span'],
                                      'mentioned': triple[entity],
                                      'mentionedSpan': triple[entity + 'Span']})
    return stringMatches

class CoreNLPService:
    def __init__(self):
       self.server = self.get_server()

    def get_server(self, host='http://localhost', port=3456, timeout=300.0):
        server = StanfordCoreNLP(host, port=port, timeout=timeout)
        return server

    def getCoref(self, text, paragraph_start_index):
        """
        Annotates coreference resolution, filtering out pairs which are both pronouns.

        :param text:
        :param paragraph_start_index:
        :return: Each coreference relation is represented by a dictionary with entries:
                'representativeMention', 'representativeMentionSpan', 'mentioned', 'mentionedSpan'.
        """

        prop = {"annotators": "coref, ssplit", 'ssplit.isOneSentence': 'true'}
        output = json.loads(self.server.annotate(text, properties=prop))['corefs']

        coref = []
        for entity in output.values():
            representativeMention = {}
            for item in entity:
                if item['isRepresentativeMention'] is True:
                    representativeMention = {'representativeMention':item['text'],
                                             'representativeMentionSpan':[paragraph_start_index + item['startIndex'] - 1,
                                                                          paragraph_start_index + item['endIndex'] - 1]}
                    break

            # Filter out pairs which are both pronouns
            if representativeMention['representativeMention'].lower() in pronouns:
                for item in entity:
                    if item['isRepresentativeMention'] is False and item['text'].lower() not in pronouns:
                        coref.append({**representativeMention, 'mentioned':item['text'],
                                             'mentionedSpan':[paragraph_start_index + item['startIndex'] - 1,
                                                                          paragraph_start_index + item['endIndex'] - 1]})

            else:
                for item in entity:
                    if item['isRepresentativeMention'] is False:
                        coref.append({**representativeMention, 'mentioned':item['text'],
                                             'mentionedSpan':[paragraph_start_index + item['startIndex'] - 1,
                                                                          paragraph_start_index + item['endIndex'] - 1]})

        return coref


    def getOpenie(self, text, paragraph_index, paragraph_start_index):
        """
        Extracts open-domain relation triples, representing a subject, a relation, and the object of the relation.

        :param text:
        :param paragraph_index:
        :param paragraph_start_index:
        :return: Each relation is represented by a dictionary with entries: "subject", "subjectSpan", "relation", "relationSpan",
                "object", "objectSpan" and "paragraphIndex".
        """
        prop = {"annotators":"openie, coref, ssplit", "openie.triple.strict":"false",
                'openie.max_entailments_per_clause':'1', 'ssplit.isOneSentence':'true'}

        triples = json.loads(self.server.annotate(text, properties=prop))['sentences'][0]['openie']

        for triple in triples:
            triple.update({'paragraphIndex':paragraph_index})
            for i in [0, 1]:
                triple['subjectSpan'][i] += paragraph_start_index
                triple['relationSpan'][i] += paragraph_start_index
                triple['objectSpan'][i] += paragraph_start_index
        return triples

def main(args):
    if args.no_raw:
        data_path = os.path.join(args.data_dir, args.split) + '.src'
    else:
        data_path = os.path.join(args.data_dir, args.split) + '.raw.src'
    texts = readfile(data_path)

    labels_path = os.path.join(args.save_dir, args.split, 'labels.txt')
    nodes_path = os.path.join(args.save_dir, args.split, 'nodes.txt')
    nodes1_path = os.path.join(args.save_dir, args.split, 'nodes1.txt')
    nodes2_path = os.path.join(args.save_dir, args.split, 'nodes2.txt')

    finished_count = _count_lines(labels_path, nodes_path, nodes1_path, nodes2_path)
    print("Processing %s from instance %d." % (data_path, finished_count + 1))
    max_lines = len(texts)
    print("Maximum %d instances." % (max_lines))

    for ind in range(finished_count, max_lines):
        text = texts[ind]
        nodes, labels, nodes1, nodes2 = buildGraphwithNE(args, text[1], ind)
        with open(nodes_path, 'a+', encoding='utf8') as f:
            f.write(' '.join(nodes))
            f.write('\n')
        with open(labels_path, 'a+', encoding='utf8') as f:
            f.write(' '.join(labels))
            f.write('\n')
        with open(nodes1_path, 'a+', encoding='utf8') as f:
            f.write(' '.join(nodes1))
            f.write('\n')
        with open(nodes2_path, 'a+', encoding='utf8') as f:
            f.write(' '.join(nodes2))
            f.write('\n')
        print('Instance %d finished.' % (ind + 1))


if __name__ == '__main__':
    args = parse_args()
    main(args)


