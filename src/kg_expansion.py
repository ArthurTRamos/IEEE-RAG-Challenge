"""Deterministic lexical linking and bounded one-hop expansion of DBpedia cards."""
import json
import math
import re
from collections import defaultdict, Counter


def tokens(text):
    return tuple(re.findall(r'[^\W_]+', text.casefold()))


class KnowledgeExpansion:
    def __init__(self, path, chunks, policy="legacy"):
        if policy not in ("legacy", "contextual"):
            raise ValueError("Unknown alias policy")
        self.policy = policy
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
        self.phrase_postings = defaultdict(set)
        self.chunk_words = {}
        for chunk in chunks:
            if self.policy == 'contextual':
                self.chunk_words[chunk['chunk_id']] = set(tokens(chunk['text']))
                for phrase, target in self.matches(chunk['text']):
                    self.phrase_postings[phrase].add(chunk['chunk_id'])
                    self.postings[target].add(chunk['chunk_id'])
            else:
                for target in self.link(chunk['text']):
                    self.postings[target].add(chunk['chunk_id'])
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

    def matches(self, text):
        words = tokens(text)
        found = set()
        for i, word in enumerate(words):
            for phrase in self.by_first.get(word, []):
                if words[i:i + len(phrase)] == phrase and len(self.names[phrase]) == 1:
                    found.add((phrase, next(iter(self.names[phrase]))))
        return sorted(found)

    def link(self, text):
        return sorted({target for _, target in self.matches(text)})

    def expand(self, question, mode):
        if self.policy == 'contextual':
            if mode not in ('control', 'aliases'):
                raise ValueError('Contextual policy is evaluated for aliases only')
            return self.expand_contextual(question, mode)
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

    def expand_contextual(self, question, mode):
        # Fixed before evaluation: >=2 context terms and >=50% coverage within
        # one supporting chunk. No question IDs, domains or relevance grades.
        stop = set('a an the of in on at to for from by with and or is are was were be been being '
                   'what which how why when where who will would could should does do did has have had '
                   'this that these those it its their they we both than as into about using use used'.split())
        words = tokens(question)
        matches = [(phrase, target) for phrase, target in self.matches(question)
                   if self.phrase_postings[phrase]] if mode != 'control' else []
        seeds = sorted({target for _, target in matches})[:8]
        mention_terms = {word for phrase, _ in matches for word in phrase}
        context = set(words) - mention_terms - stop
        context = {w for w in context if not w.isdigit()}
        additions, decisions = [], []
        seen = set(words)
        for target in seeds:
            node = self.nodes[target]
            candidates = []
            for text in sorted({node['label'], *node.get('aliases', [])}):
                key = tokens(text)
                reason, score, overlap, support = None, 0.0, 0, []
                if not key or len(key) > 8 or not any(w.isalpha() for w in key):
                    reason = 'invalid_phrase'
                elif not (set(key) - seen):
                    reason = 'already_in_query'
                elif self.names.get(key, set()) != {target}:
                    reason = 'ambiguous_or_ineligible'
                elif not self.phrase_postings[key]:
                    reason = 'no_corpus_mention'
                elif len(context) < 2:
                    reason = 'insufficient_question_context'
                else:
                    for chunk_id in sorted(self.phrase_postings[key]):
                        hits = len(context & self.chunk_words[chunk_id])
                        if hits >= 2 and hits / len(context) >= 0.5:
                            support.append(chunk_id)
                            overlap = max(overlap, hits)
                    score = overlap / len(context)
                    if not support:
                        reason = 'no_contextual_support'
                decision = {'uri': target, 'phrase': text, 'reason': reason or 'supported',
                            'context_coverage': score, 'support_chunk_ids': support}
                decisions.append(decision)
                if reason is None:
                    candidates.append((score, len(support), text, decision))
            for score, support_count, text, decision in sorted(candidates, key=lambda c: (-c[0], -c[1], c[2]))[:2]:
                novel = [w for w in tokens(text) if w not in seen]
                novel = list(dict.fromkeys(novel))
                if not novel:
                    decision['reason'] = 'already_added'
                    continue
                seen.update(novel)
                decision['reason'] = 'selected'
                additions.append({'uri': target, 'phrase': text, 'relation': 'contextual_alias',
                                  'query_terms_added': novel, 'context_coverage': score,
                                  'support_chunk_ids': decision['support_chunk_ids']})
        query = question + ''.join(' ' + ' '.join(a['query_terms_added']) for a in additions)
        return query, {'policy': 'contextual', 'seed_uris': seeds, 'additions': additions,
                       'context_terms': sorted(context), 'candidate_decisions': decisions,
                       'abstained': mode != 'control' and not additions,
                       'seed_chunk_links': {u: sorted(self.postings[u]) for u in seeds}}
