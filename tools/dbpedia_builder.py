"""Offline DBpedia construction. Standard library only; never imported by inference."""
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen

VERSION = '3.0-integrity-aliases'
NS = {'dbr': 'http://dbpedia.org/resource/', 'dbc': 'http://dbpedia.org/resource/Category:',
      'dbo': 'http://dbpedia.org/ontology/', 'dbp': 'http://dbpedia.org/property/',
      'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
      'skos': 'http://www.w3.org/2004/02/skos/core#', 'dct': 'http://purl.org/dc/terms/'}
PREFIXES = '\n'.join(f'PREFIX {p}: <{u}>' for p, u in NS.items())


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def uri(value):
    for prefix, namespace in NS.items():
        if value.startswith(prefix + ':'):
            value = namespace + value[len(prefix) + 1:]
            break
    if not value.startswith(('http://', 'https://')) or re.search(r'[\s<>"{}|^`\\]', value):
        raise ValueError(f'Invalid IRI: {value!r}')
    return value


def iri(value):
    return '<' + uri(value) + '>'


def name_from_uri(value):
    # A resource title can itself contain '/': T1/2 must stay T1/2.
    resource = NS['dbr']
    title = value[len(resource):] if value.startswith(resource) else value.rsplit('/', 1)[-1]
    return unquote(title).replace('_', ' ')


class IncompleteQuery(RuntimeError):
    pass


def http_transport(endpoint, query):
    request = Request(endpoint, data=urlencode({'query': query, 'format': 'application/sparql-results+json'}).encode(),
                      headers={'User-Agent': 'IEEE-CIS-KG-Offline/3.0',
                               'Accept': 'application/sparql-results+json'})
    with urlopen(request, timeout=60) as response:
        return response.read(), {k.lower(): v for k, v in response.headers.items()}


class Client:
    def __init__(self, config, cache, transport=http_transport, sleep=time.sleep):
        self.config, self.transport, self.sleep = config, transport, sleep
        # Publication exceptions do not change SPARQL responses; preserve the existing cache.
        query_config = {k: v for k, v in config.items() if k != 'allowed_missing_seed_labels'}
        self.namespace = digest(encode({'version': VERSION, 'config': {k: v for k, v in query_config.items() if k != 'redirect_review'}}).encode())
        self.conn = sqlite3.connect(cache)
        self.conn.execute('CREATE TABLE IF NOT EXISTS responses_v3 '
                          '(namespace TEXT, query_hash TEXT, status TEXT, response TEXT, detail TEXT, '
                          'updated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(namespace,query_hash))')
        self.failures = []
        self.events = []

    def close(self):
        self.conn.close()

    def query(self, query, retries=3):
        key = digest(query.encode())
        cached = self.conn.execute('SELECT status,response FROM responses_v3 WHERE namespace=? AND query_hash=?',
                                   (self.namespace, key)).fetchone()
        if cached and cached[0] == 'complete':
            return json.loads(cached[1])
        for attempt in range(retries):
            raw, rows, status = b'', [], 'failed'
            try:
                raw, headers = self.transport(self.config['endpoint'], query)
                headers = {k.lower(): v for k, v in headers.items()}
                result = json.loads(raw)
                rows = result['results']['bindings']
                if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                    raise ValueError('Invalid SPARQL bindings')
                status = 'partial'
                if headers.get('x-sql-state', '00000') != '00000' or headers.get('x-sql-message'):
                    raise IncompleteQuery('Endpoint reports SQL error/partial results')
                # Virtuoso can send valid JSON and indicate truncation in a header.
                for header, limit in headers.items():
                    if 'maxrows' in header or 'max-rows' in header:
                        try:
                            capped = len(rows) >= int(limit) > 0
                        except ValueError:
                            capped = True  # Unknown limit marker: do not certify completeness.
                        if capped:
                            raise IncompleteQuery('Endpoint reports row limit')
                self._save(key, 'complete', rows, '')
                self.sleep(0.15)
                return rows
            except Exception as error:
                detail = str(error)[:1000]
                if raw and (b'S1TAT' in raw or b'"bindings"' in raw):
                    status = 'partial'
                self._save(key, status, rows, detail)
                self.events.append({'query_hash': key, 'status': status, 'detail': detail})
                if attempt + 1 == retries:
                    raise IncompleteQuery(detail) from error
                self.sleep(2 ** attempt)
        raise IncompleteQuery('No attempts configured')

    def _save(self, key, status, rows, detail):
        with self.conn:
            self.conn.execute('INSERT OR REPLACE INTO responses_v3 '
                              '(namespace,query_hash,status,response,detail) VALUES (?,?,?,?,?)',
                              (self.namespace, key, status, encode(rows), detail))

    def paged(self, body):
        """SELECT DISTINCT, stable ORDER BY all projected variables, bounded OFFSET pages.

        Live endpoint mutations and unadvertised server caps cannot be excluded.
        Use a fixed snapshot for reproducibility across collection dates.
        """
        variables = re.search(r'SELECT\s+DISTINCT\s+(.+?)\s+WHERE', body, re.I | re.S)
        if not variables:
            raise ValueError('Expected SELECT DISTINCT with explicit projected variables')
        order = ' '.join(re.findall(r'\?\w+', variables.group(1)))
        size = self.config['page_size']
        rows, seen_pages = [], set()
        for page in range(self.config['max_pages']):
            query = f'{PREFIXES}\n{body}\nORDER BY {order}\nLIMIT {size} OFFSET {page * size}'
            batch = self.query(query)
            if len(batch) > size:
                raise IncompleteQuery('Endpoint ignored page size')
            fingerprint = digest(encode(batch).encode())
            if batch and fingerprint in seen_pages:
                raise IncompleteQuery('Endpoint repeated a page')
            seen_pages.add(fingerprint)
            rows.extend(batch)
            if len(batch) < size:
                return sorted({encode(r): r for r in rows}.values(), key=encode)
        raise IncompleteQuery('Pagination budget exhausted')

    def batch(self, values, make_query):
        values = sorted(set(values))
        try:
            return self.paged(make_query(values)) if values else []
        except IncompleteQuery as error:
            if len(values) == 1:
                self.failures.append({'uri': values[0], 'query_hash': digest(make_query(values).encode()),
                                      'status': 'failed', 'detail': str(error)})
                raise  # Never silently skip a URI or publish an incomplete build.
            middle = len(values) // 2
            return self.batch(values[:middle], make_query) + self.batch(values[middle:], make_query)

    def collect(self, values, make_query, size=None):
        values = sorted(set(values))
        rows = []
        size = size or self.config['batch_size']
        for start in range(0, len(values), size):
            rows.extend(self.batch(values[start:start + size], make_query))
        return sorted({encode(r): r for r in rows}.values(), key=encode)


def values(items):
    return ' '.join(iri(u) for u in sorted(set(items)))


def resolve_seeds(client, seeds):
    reports = []
    for requested in sorted({uri(s) for s in seeds}):
        chain, current, status = [requested], requested, None
        for _ in range(client.config['max_redirect_hops']):
            rows = client.batch([current], lambda batch: 'SELECT DISTINCT ?target WHERE { '
                                f'VALUES ?source {{ {values(batch)} }} ?source dbo:wikiPageRedirects ?target . }}')
            targets = sorted({r['target']['value'] for r in rows})
            if not targets:
                status = 'resolved' if len(chain) > 1 else 'unchanged'
                break
            if len(targets) != 1:
                status = 'ambiguous_redirect'
                break
            target = uri(targets[0])
            if target in chain:
                status = 'redirect_cycle'
                break
            chain.append(target)
            current = target
        reports.append({'requested_uri': requested, 'canonical_uri': current,
                        'chain': chain, 'status': status or 'hop_limit'})
    return reports


def name_record(text, kind, source_uri, language='en', **extra):
    return {'text': text, 'kind': kind, 'language': language, 'source_uri': source_uri, **extra}


def alias_index(cards):
    index = {}
    for card in cards:
        for record in card['names']:
            index.setdefault(record['text'].casefold(), set()).add(card['uri'])
    return {name: sorted(targets) for name, targets in sorted(index.items())}


def extract(client):
    cfg = client.config
    seeds = resolve_seeds(client, cfg['SEED_ENTITIES'])
    invalid = [s for s in seeds if s['status'] not in ('resolved', 'unchanged')]
    if invalid:
        raise IncompleteQuery('Unresolved seeds: ' + encode(invalid))
    categories = {uri(c) for c in cfg['SEED_CATEGORIES']}
    frontier, skos = set(categories), set()
    for _ in range(cfg['depth']):
        rows = client.collect(frontier, lambda b: 'SELECT DISTINCT ?sub ?parent WHERE { '
                              f'VALUES ?parent {{ {values(b)} }} ?sub skos:broader ?parent . }}')
        next_frontier = set()
        for row in rows:
            sub, parent = row['sub']['value'], row['parent']['value']
            if any(word in sub.lower() for word in cfg['EXCLUDED_CAT_KEYWORDS']):
                continue
            skos.add((sub, parent))
            if sub not in categories:
                next_frontier.add(sub)
                categories.add(sub)
        frontier = next_frontier
    entities = {s['canonical_uri'] for s in seeds}
    subjects = set()
    rows = client.collect(categories, lambda b: 'SELECT DISTINCT ?entity ?cat WHERE { '
                          f'VALUES ?cat {{ {values(b)} }} ?entity dct:subject ?cat . '
                          'FILTER NOT EXISTS { ?entity a dbo:Person } '
                          'FILTER NOT EXISTS { ?entity a dbo:FictionalCharacter } '
                          'FILTER NOT EXISTS { ?entity a dbo:Organisation } }')
    for row in rows:
        ent, cat = row['entity']['value'], row['cat']['value']
        if any(ent.startswith(p) for p in cfg['INVALID_RESOURCE_PREFIXES']):
            continue
        if any(bad in ent.lower() for bad in ('_(comics)', '_(film)', '_(character)', '_(novel)', '_(band)')):
            continue
        entities.add(ent)
        subjects.add((ent, cat))
    metadata = client.collect(entities, lambda b: 'SELECT DISTINCT ?entity ?label ?comment WHERE { '
                              f'VALUES ?entity {{ {values(b)} }} ?entity rdfs:label ?label . '
                              'FILTER(LANG(?label)="en") OPTIONAL { ?entity rdfs:comment ?comment . '
                              'FILTER(LANG(?comment)="en") } }')
    cards = {}
    for row in metadata:
        ent = row['entity']['value']
        label, comment = row['label']['value'], row.get('comment', {}).get('value', '')
        if len(comment) > cfg['max_comment_chars']:
            comment = comment[:cfg['max_comment_chars']].rsplit(' ', 1)[0] + '...'
        record = cards.setdefault(ent, {'uri': ent, 'label': label, 'definition': comment,
                                       'names': [], 'aliases': [], 'categories': [],
                                       'properties': [], 'related_concepts': []})
        record['names'].append(name_record(label, 'label', ent, row['label'].get('xml:lang', '')))
    for seed in seeds:
        seed['metadata_status'] = 'present' if seed['canonical_uri'] in cards else 'missing_label'
    allowed = cfg.get('allowed_missing_seed_labels', {})
    if any(not isinstance(reason, str) or not reason.strip() for reason in allowed.values()):
        raise ValueError('Missing-seed exceptions require a nonempty reason')
    missing = []
    for seed in seeds:
        if seed['metadata_status'] == 'missing_label':
            reason = allowed.get(seed['requested_uri'])
            if reason:
                seed['metadata_status'] = 'excluded_missing_label'
                seed['exclusion_reason'] = reason
            else:
                missing.append(seed)
    if missing:
        raise IncompleteQuery('Seeds without labels: ' + encode(missing))
    redirects = client.collect(cards, lambda b: 'SELECT DISTINCT ?entity ?redirect WHERE { '
                               f'VALUES ?entity {{ {values(b)} }} ?redirect dbo:wikiPageRedirects ?entity . }}')
    for row in redirects:
        ent, redirect = row['entity']['value'], row['redirect']['value']
        text = name_from_uri(redirect)
        if text.strip():
            cards[ent]['names'].append(name_record(text, 'redirect', redirect, ''))
    # Also preserve original seed names across redirect chains.
    for seed in seeds:
        if seed['canonical_uri'] not in cards:
            continue
        for source in seed['chain'][:-1]:
            cards[seed['canonical_uri']]['names'].append(name_record(name_from_uri(source), 'seed_redirect', source, ''))
    for alias in cfg.get('reviewed_aliases', []):
        # Explicitly sourced global aliases only. Document-local acronyms belong in
        # the runtime corpus index, never in this global DBpedia dictionary.
        if not alias.get('text', '').strip() or not alias.get('source', '').strip():
            raise ValueError('Reviewed aliases require text and source')
        target = uri(alias['canonical_uri'])
        if target not in cards:
            raise ValueError('Reviewed alias target is absent: ' + target)
        cards[target]['names'].append(name_record(alias['text'], 'reviewed', alias['source'],
                                                  alias.get('language', 'en')))
    properties = client.collect(cards, lambda b: 'SELECT DISTINCT ?entity ?prop ?val WHERE { '
                                f'VALUES ?entity {{ {values(b)} }} VALUES ?prop {{ {values(uri(p) for p in cfg["TARGET_PROPERTIES"])} }} '
                                '?entity ?prop ?val . FILTER(!isLiteral(?val) || LANG(?val)="" || LANG(?val)="en") }')
    links = client.collect(cards, lambda b: 'SELECT DISTINCT ?src ?dst WHERE { '
                           f'VALUES ?src {{ {values(b)} }} ?src dbo:wikiPageWikiLink ?dst . }}',
                           cfg['wikilink_batch_size'])
    # Structured bindings remain in the collection; display properties stay compatible.
    triples = set()
    for sub, parent in sorted(skos):
        triples.add((sub, uri('skos:broader'), iri(parent)))
        triples.add((sub, uri('rdfs:label'), encode(name_from_uri(sub).removeprefix('Category:')) + '@en'))
    for ent, cat in sorted(subjects):
        if ent in cards:
            cards[ent]['categories'].append(name_from_uri(cat).removeprefix('Category:'))
            triples.add((ent, uri('dct:subject'), iri(cat)))
    edges = set()
    for row in links:
        src, dst = row['src']['value'], row['dst']['value']
        if src in cards and dst in cards and src != dst:
            edges.add((src, dst))
            triples.add((src, uri('dbo:wikiPageWikiLink'), iri(dst)))
            cards[src]['related_concepts'].append(cards[dst]['label'])
    for row in properties:
        ent, prop, val = row['entity']['value'], row['prop']['value'], row['val']
        triples.add((ent, prop, rdf_value(val)))
        display = name_from_uri(val['value']) if val['type'] == 'uri' else val['value']
        cards[ent]['properties'].append(prop.rsplit('/', 1)[-1] + ': ' + display)
    for ent, card in sorted(cards.items()):
        for field in ('categories', 'properties', 'related_concepts'):
            card[field] = sorted(set(card[field]))  # No arbitrary construction-time truncation.
        card['names'] = sorted({encode(n): n for n in card['names']}.values(), key=encode)
        card['aliases'] = sorted({n['text'] for n in card['names'] if n['text'] != card['label']})
        for n in card['names']:
            if n['kind'] == 'label':
                triples.add((ent, uri('rdfs:label'), rdf_value({'type': 'literal', 'value': n['text'], 'xml:lang': n['language']})))
        if card['definition']:
            triples.add((ent, uri('rdfs:comment'), encode(card['definition']) + '@en'))
    return {'cards': [cards[u] for u in sorted(cards)], 'triples': sorted(triples),
            'seeds': seeds, 'missing_metadata_uris': sorted(entities - cards.keys()),
            'edges': [{'source_uri': s, 'predicate_uri': uri('dbo:wikiPageWikiLink'), 'target_uri': d}
                      for s, d in sorted(edges)]}


def rdf_value(binding):
    if binding['type'] == 'uri':
        return iri(binding['value'])
    if binding['type'] not in ('literal', 'typed-literal'):
        raise ValueError('Unsupported RDF binding')
    result = encode(binding['value'])
    if binding.get('xml:lang'):
        if not re.fullmatch(r'[a-zA-Z]+(?:-[a-zA-Z0-9]+)*', binding['xml:lang']):
            raise ValueError('Invalid language tag')
        return result + '@' + binding['xml:lang']
    return result + ('^^' + iri(binding['datatype']) if binding.get('datatype') else '')


def apply_redirect_review(data, review):
    """Apply explicit reviewed exceptions, without silently equating related concepts."""
    data = json.loads(json.dumps(data))
    blocked = {(uri(r['source_uri']), uri(r['target_uri'])): r['reason']
               for r in review.get('rejected_equivalences', [])}
    excluded = {uri(r['uri']): r['reason'] for r in review.get('excluded_entities', [])}
    if any(not reason.strip() for reason in [*blocked.values(), *excluded.values()]):
        raise ValueError('Redirect review requires reasons')
    changes = []
    cards = {c['uri']: c for c in data['cards'] if c['uri'] not in excluded}
    for ent, card in sorted(cards.items()):
        names = []
        for record in card['names']:
            if record['kind'] in ('redirect', 'seed_redirect'):
                if (record['source_uri'], ent) in blocked:
                    changes.append({'action': 'reject_equivalence', 'target_uri': ent, **record})
                    continue
                corrected = name_from_uri(record['source_uri'])
                if corrected != record['text']:
                    changes.append({'action': 'repair_name', 'old_text': record['text'], 'new_text': corrected,
                                    'source_uri': record['source_uri'], 'target_uri': ent})
                record['text'] = corrected
            names.append(record)
        card['names'] = sorted({encode(n): n for n in names}.values(), key=encode)
        card['aliases'] = sorted({n['text'] for n in names if n['text'] != card['label']})
    data['cards'] = [cards[u] for u in sorted(cards)]
    data['edges'] = [e for e in data['edges'] if e['source_uri'] in cards and e['target_uri'] in cards]
    neighbors = {}
    for edge in data['edges']:
        neighbors.setdefault(edge['source_uri'], set()).add(cards[edge['target_uri']]['label'])
    for ent, card in cards.items():
        card['related_concepts'] = sorted(neighbors.get(ent, set()))
    data['triples'] = [t for t in data['triples'] if t[0] not in excluded and t[2] not in {iri(u) for u in excluded}]
    for seed in data['seeds']:
        requested, canonical = seed['requested_uri'], seed['canonical_uri']
        if (requested, canonical) in blocked:
            seed['endpoint_canonical_uri'] = canonical
            seed['review_reason'] = blocked[(requested, canonical)]
            if requested in cards:
                seed['canonical_uri'] = requested
                seed['status'] = 'reviewed_distinct_resource'
                seed['metadata_status'] = 'present'
            else:
                seed['metadata_status'] = 'excluded_redirect_review'
        elif canonical in excluded:
            seed['metadata_status'] = 'excluded_redirect_review'
            seed['exclusion_reason'] = excluded[canonical]
    data['derivation'] = {**data.get('derivation', {}), 'redirect_review': review,
                          'review_changes': changes, 'excluded_entities': excluded}
    return data


def publish(output, data, config, namespace):
    """Publish a new immutable directory by a single rename, or publish nothing."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.dbpedia-', dir=output.parent))
    try:
        cards = data['cards']
        cards = sorted(cards, key=lambda c: c['uri'])
        if not cards or len({c['uri'] for c in cards}) != len(cards):
            raise ValueError('Empty or duplicate card inventory')
        for card in cards:
            uri(card['uri'])
            if not card['label']:
                raise ValueError('Empty label')
        (staging / 'kg_rag_chunks.jsonl').write_text(''.join(encode(c) + '\n' for c in cards), encoding='utf-8')
        # N-Triples is a deterministic subset of Turtle. All terms are validated on construction.
        (staging / 'domain_kg.ttl').write_text(''.join(f'{iri(s)} {iri(p)} {o} .\n' for s, p, o in sorted(data['triples'])), encoding='utf-8')
        (staging / 'edges.jsonl').write_text(''.join(encode(e) + '\n' for e in data['edges']), encoding='utf-8')
        (staging / 'alias_index.json').write_text(encode(alias_index(cards)) + '\n', encoding='utf-8')
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p.read_bytes())} for p in sorted(staging.iterdir())}
        manifest = {'status': 'complete', 'builder_version': VERSION, 'cache_namespace': namespace,
                    'snapshot_caveat': 'Live endpoint cannot guarantee reproducibility across dates or unadvertised truncation.',
                    'config': config, 'seeds': data['seeds'], 'missing_metadata_uris': data['missing_metadata_uris'],
                    'derivation': data.get('derivation'),
                    'counts': {'cards': len(cards), 'triples': len(data['triples']), 'edges': len(data['edges'])}, 'files': files}
        (staging / 'manifest.json').write_text(encode(manifest) + '\n', encoding='utf-8')
        validate_bundle(staging)
        if sum(p.stat().st_size for p in staging.iterdir()) > config['max_bundle_bytes']:
            raise ValueError('Bundle exceeds configured budget; no files published (no automatic pruning)')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def validate_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Incomplete bundle')
    for name, expected in manifest['files'].items():
        raw = (directory / name).read_bytes()
        if len(raw) != expected['bytes'] or digest(raw) != expected['sha256']:
            raise ValueError('Artifact hash mismatch: ' + name)
    cards = [json.loads(line) for line in (directory / 'kg_rag_chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(cards) != manifest['counts']['cards'] or len({c['uri'] for c in cards}) != len(cards):
        raise ValueError('Invalid card count')
    if json.loads((directory / 'alias_index.json').read_text(encoding='utf-8')) != alias_index(cards):
        raise ValueError('Alias index mismatch')
    triples = (directory / 'domain_kg.ttl').read_text(encoding='utf-8').splitlines()
    if len(triples) != manifest['counts']['triples']:
        raise ValueError('Invalid triple count')
    # Match the emitted Turtle subset, including escaped literals, languages and datatypes.
    triple_pattern = r'<[^<>\s]+> <[^<>\s]+> (?:<[^<>\s]+>|"(?:[^"\\\n]|\\.)*"(?:@[A-Za-z0-9-]+|\^\^<[^<>\s]+>)?) \.'
    if any(not re.fullmatch(triple_pattern, line) for line in triples):
        raise ValueError('Invalid Turtle term structure')
    triple_set = set(triples)
    if len(triple_set) != len(triples):
        raise ValueError('Duplicate RDF triples')
    inventory = {c['uri'] for c in cards}
    for card in cards:
        expected_label = f'{iri(card["uri"])} {iri(uri("rdfs:label"))} {encode(card["label"])}@en .'
        if expected_label not in triple_set:
            raise ValueError('Card label missing from Turtle')
    edges = [json.loads(line) for line in (directory / 'edges.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(edges) != manifest['counts']['edges']:
        raise ValueError('Invalid edge count')
    for edge in edges:
        if edge['source_uri'] not in inventory or edge['target_uri'] not in inventory:
            raise ValueError('Edge outside card inventory')
        expected_edge = f'{iri(edge["source_uri"])} {iri(edge["predicate_uri"])} {iri(edge["target_uri"])} .'
        if expected_edge not in triple_set:
            raise ValueError('Edge missing from Turtle')
    return manifest


def build(config, cache, output, transport=http_transport, sleep=time.sleep):
    if Path(output).exists():
        raise FileExistsError(output)
    if any(config.get(k, 0) <= 0 for k in ('page_size', 'max_pages', 'batch_size', 'wikilink_batch_size', 'max_redirect_hops', 'max_bundle_bytes')):
        raise ValueError('Limits must be positive')
    client = Client(config, cache, transport, sleep)
    try:
        data = extract(client)
        if config.get('redirect_review'):
            data = apply_redirect_review(data, config['redirect_review'])
        return publish(output, data, config, client.namespace)
    except Exception as error:
        # Separate diagnostics never masquerade as the successful output directory.
        report = Path(str(output) + '.failed.json')
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(encode({'status': 'failed', 'error': str(error), 'cache_namespace': client.namespace,
                                  'failures': client.failures, 'events': client.events}) + '\n', encoding='utf-8')
        raise
    finally:
        client.close()
