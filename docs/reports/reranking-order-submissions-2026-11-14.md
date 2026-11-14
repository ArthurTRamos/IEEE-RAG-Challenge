# Duas submissões: seleção híbrida e reordenação do contexto

## Escolha e ordem sugerida

1. **MiniLM corrigido, 512 tokens:** primeira opção experimental. A comparação
   local com OLMo 7B mostrou melhorias pontuais em relação à ordem híbrida,
   especialmente em vazão, bypass, volume e definição de softmax, sem provar
   acurácia superior no ambiente oficial. Preservar a seleção evita retirar
   evidências do híbrido, cuja melhor acurácia informada é 60%.
2. **Ettin 68M, 1.024 tokens:** segunda opção para testar outra configuração
   de reordenação sobre o mesmo conjunto. O proxy@10 local foi um pouco maior
   que o MiniLM. A janela maior reduziu truncamentos de pares e corrigiu a
   escala da resposta de volume em relação ao Ettin 512 na avaliação com 7B.
   Os resultados de QA foram mistos e não demonstraram superioridade global;
   seu custo CPU observado foi maior. É uma submissão informativa, não uma
   melhoria garantida.

Os dois pacotes comparam configurações de modelo e janela escolhidas entre
as avaliadas. Não isolam estritamente arquitetura, pois os limites e tokenizers
também diferem. O aspecto controlado é manter a mesma recuperação e o mesmo
conjunto de evidências que o híbrido enviaria à LLM. Uma comparação de janela
Ettin 512/1.024 oficial demandaria outra submissão.

Não escolhemos repetir o reranking anterior à seleção: esse desenho já obteve
55% informado, contra 60% do híbrido. Corrigir somente o truncamento da pergunta
alterou significativamente apenas a questão 13 na amostra pública. Os novos
pacotes testam a ordenação sem a substituição do conjunto selecionado.

## Comportamento implementado

`retrieve(..., rerank_mode="selected_context")`:

1. Executa MiniLM denso + BM25 k1=1,25/b=0,75 e RRF 60, pesos 1/0,75.
2. Seleciona o contexto híbrido pelo orçamento original de 10.000 tokens de
   texto dos chunks, parando no primeiro overflow.
3. Pontua os primeiros 100 candidatos do ranking híbrido com o CE. Preserva a
   pergunta inteira, truncando somente a passagem para a pontuação.
4. Reordena o ranking pontuado e preserva a cauda híbrida; filtra essa ordem
   para permutar exatamente os chunks já selecionados.
5. Remonta os rótulos Fragment na nova ordem, sem repacking e sem substituir
   o texto de um chunk que tenha sido truncado pelo fallback original.

Se a pergunta ocupar a janela do CE, seu fallback preserva a ordem híbrida e
o contexto original. Scores CE e RRF não são somados. Os registros selecionados
retêm a posição de origem em `rank`; a posição na lista define a ordem final.

Não há um limite fixo de 50 chunks para a LLM. Os 100 são o pool de pontuação
do reranker; a seleção mantém o orçamento do baseline. O texto do prompt, o
modelo OLMo fornecido, BF16/CUDA, geração greedy e limite de 2.048 tokens de
saída permanecem conforme o baseline. Modelos CE rodam em CPU e são carregados
uma vez por processo. As threads de inferência seguem o ambiente; as validações
locais usaram explicitamente quatro threads.

O `main.py` do repositório mantém o CE desativado. O construtor ativa o modo e
a janela apenas no `main.py` de cada ZIP. Os pacotes históricos foram preservados.

## Pacotes e manifestos

| Variante | ZIP | Bytes |
| --- | --- | ---: |
| MiniLM 512 | `2026-11-14-hybrid-selected-minilm-top100-q512-order-only.zip` | 482.432.626 |
| Ettin 1.024 | `2026-11-14-hybrid-selected-ettin68m-top100-q1024-order-only.zip` | 653.427.898 |

Ambos ficam abaixo de 1 GB. O nome `q512`/`q1024` identifica a janela conjunta
do par no CE, não um corte aplicado à pergunta isoladamente. Os JSONs em
`experiments/submissions/` têm os mesmos nomes-base e registram configuração,
revisão dos modelos, SHA-256 do ZIP e de cada entrada, run de referência e
validações. O Ettin inclui todos os módulos da cabeça treinada.

SHA-256 MiniLM:
`7eec2dd9ddc77c33564d31cb5b7a087ce6577871328d365402af2c2024fec7a6`.

SHA-256 Ettin:
`b198022079338a9a5e729ebd88f6cd3534bcdaf9fb494864bfaddbfd64013cf1`.

## Validação e limites

- Passaram **54 testes**, incluindo seleção antes de reordenar, preservação
  do texto truncado do primeiro chunk, fallback idêntico ao baseline e modo legado.
- CRC, tamanho e hashes de todas as entradas foram conferidos; modelos locais
  foram verificados contra suas proveniências antes de empacotar.
- Os **dois ZIPs extraídos** reproduziram byte a byte os contextos das 22 perguntas
  de suas respectivas ablações, usando scores CE e densos congelados e BM25 real.
- `answer(question, corpus_dir, model_id)` foi exercitado offline em cada ZIP
  com CE real pontuando 100 candidatos, tokenizer OLMo real, template de chat
  real e gerador substituto. A mensagem enviada coincidiu com a referência.
- O mesmo `answer()` passou com pergunta longa demais, sem chamar o CE e
  preservando o contexto híbrido. O reranker foi conferido como reutilizado.

Não houve inferência completa com OLMo 32B ou teste total na H100. Extração
Docling e embeddings foram reutilizados dos artefatos congelados na validação;
os arquivos locais necessários estão incluídos e tiveram seus hashes conferidos.
Não foram acrescentadas dependências ou recursos externos de consulta.

O Ettin tem maior risco de tempo. As observações anteriores de CPU/quatro
threads foram 86,31 s para MiniLM e 363,53 s para Ettin 1.024 em 2.200 pares;
não validam o custo total oficial, nem devem ser extrapoladas diretamente.
Não afirmar que a execução completa cabe em 60 minutos. O tempo deve ser
acompanhado na avaliação, principalmente na segunda submissão.

Nenhum pacote foi enviado à plataforma. Os nomes contêm a data de construção;
não definem os dias em que serão submetidos.

## Reprodução

```bash
python3 tools/package_reranker_submission.py --variant minilm-selected \
  --output /tmp/minilm-selected.zip --manifest /tmp/minilm-selected.json

python3 tools/package_reranker_submission.py --variant ettin1024-selected \
  --output /tmp/ettin-selected.zip --manifest /tmp/ettin-selected.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tools/validate_reranker_submission.py \
  --manifest /tmp/minilm-selected.json

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tools/validate_reranker_submission.py \
  --manifest /tmp/ettin-selected.json
```

O construtor recusa sobrescrever arquivos. A validação atualiza apenas o
manifesto; não altera o ZIP. O modo `minilm-before` segue disponível para o
comportamento anterior à seleção, mas não é um dos dois pacotes recomendados.
