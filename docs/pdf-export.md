# Exportar PDFs e chunks em Markdown

Na raiz do projeto, com as dependências oficiais e o modelo local do Docling:

```bash
.venv/bin/python tools/export_pdf_chunks.py --device cuda
```

Use `--device cpu` para CPU ou `--device auto` para seleção automática (padrão).
O script não instala dependências nem baixa modelos. A GPU precisa estar
acessível ao processo. `--output-dir` permite escolher outra pasta de saída;
a pasta deve estar vazia para preservar resultados anteriores.
Para reprocessar um documento isoladamente, use `--document-id AI-029` com
outra pasta de saída. A opção pode ser repetida.
Se o parser paralelo do Docling retornar uma conversão parcial, a opção
`--pdf-backend serial-docling-parse` permite tentar o parser serial do próprio
Docling. Essa escolha é registrada no manifesto; não altera `src/`.
`--pdf-backend pypdfium2` seleciona o backend PDFium do Docling como alternativa
para arquivos que o parser padrão não consegue ler.

A saída padrão é `experiments/runs/pdf-markdown-chunks/`:

- `markdown/AI-001.md`: texto completo extraído do documento.
- `chunks/AI-001/chunk-0000.md`: um arquivo por chunk, numerado a partir de zero.
- `manifest.json`: configuração, versões, hashes dos PDFs e arquivos gerados,
  metadados históricos, IDs dos chunks, tempos, recursos e falhas.

Todos os PDFs de `data/pdfs/` são descobertos recursivamente e seus hashes são
conferidos com `experiments/judments/public-22-questions.json`. Os graus de
relevância não filtram os documentos nem alteram a extração. Os metadados
históricos ficam separados do texto; caminhos históricos dentro deles não
representam os caminhos dos arquivos recém-gerados.

O exportador reutiliza `_export_markdown` e `_chunk_markdown` de
`src/retrieval.py`: Docling sem OCR/estrutura de tabelas, reparos locais com
pdfplumber, alvo de 800 caracteres e sobreposição de 150. O tamanho não é um
limite rígido: parágrafos longos são preservados. Os chunks são calculados
lendo os Markdown salvos e não recebem cabeçalhos adicionais.

Não são carregados embeddings nem o LLM. A interface de submissão e seu pacote
permanecem iguais; o script e as saídas são ferramentas locais e não entram no
ZIP. O tempo desta exportação não mede o pipeline completo de 60 minutos.

## Execução de 19/11/2026

Foram consolidados 64 Markdown e 5.156 chunks na pasta padrão, usando a RTX
4050. `AI-029` precisou do backend `pypdfium2`: o parser padrão encontrou uma
referência a uma imagem ausente na página 13. PDFium concluiu as 36 páginas;
o PDF original foi preservado. Os demais documentos usam o backend padrão.
Essa diferença é registrada em `extraction_config` por documento no manifesto.

`runs/attempt-*.json` preserva os manifestos das quatro tentativas. As falhas
iniciais constam no histórico; o manifesto consolidado tem estado `complete`.
`validation.json` registra a conferência de hashes, contagens de páginas,
IDs e correspondência exata entre os chunks e a função de `src/` aplicada aos
Markdown salvos. Os nove testes de `tests/test_retrieval.py` também passaram.
Essas verificações não constituem avaliação semântica da qualidade da extração.
