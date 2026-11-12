"""Section annotations and document-grade proxies; no passage relevance claims."""
from bisect import bisect_right
from collections import Counter, defaultdict
import re


def section_type(title):
    value = re.sub(r'^\s*[\d.IVX]+[.)\s]+', '', title, flags=re.I).casefold()
    for name, pattern in (
        ('references', r'references|bibliography|literature cited|referências'),
        ('abstract', r'abstract|resumo'), ('introduction', r'introduction|introdução|background'),
        ('conclusion', r'conclusion|concluding|conclusão'),
        ('methods', r'method|methodology|experimental|materials|procedure'),
        ('results', r'results|evaluation|experiments'), ('discussion', r'discussion'),
        ('appendix', r'appendix|appendices|supplement'),
        ('acknowledgments', r'acknowledg|funding|declaration')):
        if re.search(pattern, value):
            return name
    return 'other' if title else 'unknown'


def annotate_sections(chunks, overlap=150):
    """Reassemble normalized exported Markdown, checking every overlap exactly.

    Primary section is assigned by largest character span. Mixed chunks retain
    all contributing section titles. No text/ID/ranking modification occurs.
    """
    documents = defaultdict(list)
    for chunk in chunks:
        documents[chunk['document_id']].append(chunk)
    result = {}
    for rows in documents.values():
        rows.sort(key=lambda c: int(c['chunk_id'].rsplit('#', 1)[1]))
        markdown, intervals = '', []
        for i, chunk in enumerate(rows):
            text = chunk['text']
            tail = min(overlap, len(rows[i-1]['text'])) if i else 0
            if tail and text[:tail] != markdown[-tail:]:
                raise ValueError(f"Chunk overlap mismatch: {chunk['chunk_id']}")
            start = len(markdown) - tail
            markdown += text[tail:]
            intervals.append((start, len(markdown), chunk))
        headings = [(0, '')]
        headings.extend((m.start(), m.group(1).strip()) for m in
                        re.finditer(r'^#{1,6}\s+(.+)$', markdown, re.M))
        positions = [p for p, _ in headings]
        for start, end, chunk in intervals:
            index = bisect_right(positions, start) - 1
            spans = Counter()
            while index < len(headings) and headings[index][0] < end:
                stop = positions[index + 1] if index + 1 < len(positions) else len(markdown)
                spans[headings[index][1]] += max(0, min(end, stop) - max(start, positions[index]))
                index += 1
            title = max(spans, key=spans.get)
            result[chunk['chunk_id']] = {'section_title': title, 'section_type': section_type(title),
                'section_spans_characters': dict(spans), 'mixed_sections': len(spans) > 1,
                'annotation_source': 'Docling Markdown headings reconstructed with verified overlap'}
    return result


def context_diagnostics(selected, qrels, annotations):
    grades, sections, cross = Counter(), Counter(), Counter()
    for chunk in selected:
        grade = str(qrels[chunk['document_id']])
        section = annotations[chunk['chunk_id']]['section_type']
        grades[grade] += 1
        sections[section] += 1
        cross[f'{section}|grade_{grade}'] += 1
    size = len(selected)
    relevant = sum(grades[str(g)] for g in (1, 2))
    return {'selected_chunks': size, 'by_document_grade': dict(grades),
            'fraction_from_grade2_documents': grades['2'] / size if size else None,
            'fraction_from_grade1_or2_documents': relevant / size if size else None,
            'by_primary_section': dict(sections), 'section_by_document_grade': dict(cross),
            'mixed_section_chunks': sum(annotations[c['chunk_id']]['mixed_sections'] for c in selected),
            'truncated_chunks': sum(c.get('truncated', False) for c in selected)}
