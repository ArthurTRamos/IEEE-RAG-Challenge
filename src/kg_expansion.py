"""Deterministic lexical linking and bounded one-hop expansion of DBpedia cards."""
import json
import math
import re
from collections import defaultdict, Counter


def tokens(text):
    return tuple(re.findall(r'[^\W_]+', text.casefold()))


class KnowledgeExpansion:
    def __init__(self, path, chunks):
        self.nodes = {}
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['uri'] in self.nodes:
                raise ValueError('Duplicate KG URI')
            self.nodes[row['uri']] = row
        self.names = defaultdict(set)
        self.labels = defaultdict(set)
        for uri, node in sorted(self.nodes.items()):
            self.labels[node['label']].add(uri)
            for name in [node['label'], *node.get('aliases', [])]:
                key = tokens(name)
                if key and (len(key) > 1 or len(key[0]) >= 4):
                    self.names[key].add(uri)
        self.by_first = defaultdict(list)
        for phrase in self.names:
            self.by_first[phrase[0]].append(phrase)
        self.postings = defaultdict(set)
        for chunk in chunks:
            for uri in self.link(chunk['text']):
                self.postings[uri].add(chunk['chunk_id'])
        self.edges = defaultdict(set)
        self.ambiguous_edges = 0
        for uri, node in sorted(self.nodes.items()):
            for label in node.get('related_concepts', []):
                targets = self.labels.get(label, set())
                if len(targets) == 1:
                    target = next(iter(targets))
                    if target != uri:
                        self.edges[uri].add(target)
                elif len(targets) > 1:
                    self.ambiguous_edges += 1

    def link(self, text):
        words = tokens(text)
        found = set()
        for i, word in enumerate(words):
            for phrase in self.by_first.get(word, []):
                if words[i:i + len(phrase)] == phrase and len(self.names[phrase]) == 1:
                    found.update(self.names[phrase])
        return sorted(found)

    def expand(self, question, mode):
        if mode not in ('control', 'aliases', 'one_hop'):
            raise ValueError('Unknown KG expansion mode')
        seeds = [u for u in self.link(question) if self.postings[u]][:8] if mode != 'control' else []
        additions, seen = [], {tokens(question)}
        def add(uri, text, relation):
            key = tokens(text)
            if key and key not in seen and len(key) <= 8:
                seen.add(key)
                additions.append({'uri': uri, 'phrase': text, 'relation': relation})
        for uri in seeds:
            node = self.nodes[uri]
            add(uri, node['label'], 'seed_label')
            for alias in sorted(set(node.get('aliases', [])))[:2]:
                add(uri, alias, 'alias')
        if mode == 'one_hop':
            hits = Counter(v for u in seeds for v in self.edges[u]
                           if v not in seeds and self.postings[v])
            neighbors = sorted(hits, key=lambda u: (-hits[u] / math.log2(2 + len(self.edges[u])), u))[:8]
            for uri in neighbors:
                add(uri, self.nodes[uri]['label'], 'wikiPageWikiLink')
        query = question + ''.join(' ' + a['phrase'] for a in additions)
        return query, {'seed_uris': seeds, 'additions': additions,
                       'seed_chunk_links': {u: sorted(self.postings[u]) for u in seeds}}
