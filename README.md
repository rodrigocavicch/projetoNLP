# Reprodução — Sentiment Analysis of Twitter's Opinion on The Russia and Ukraine War Using BERT

Reprodução experimental (Modalidade 1) do artigo:

> Julianto, M. F., Malau, Y., & Hidayat, W. F. (2022). **Sentiment Analysis of Twitter's Opinion on The Russia and Ukraine War Using Bert**. *Jurnal Riset Informatika*, 5(1), 15–24. DOI: [10.34288/jri.v5i1.169](https://doi.org/10.34288/jri.v5i1.169)

Trabalho da disciplina de **Processamento de Linguagem Natural**.

**Equipe:** Artur de Paola Prieto Carvalho, Bruno Ferreira Balieiro, Pedro Ernesto Pedreira Bitencourt, Rodrigo Delphino Cavicchioli.

## Visão geral

O artigo original aplica fine-tuning do **BERT-base Multilingual Cased** à classificação
de sentimento (negativo / neutro / positivo) de 17.951 tweets sobre a guerra
Rússia–Ucrânia (março/2022), reportando **97% de acurácia**. Este repositório
reproduz o pipeline completo — limpeza → rotulação → tokenização → fine-tuning →
avaliação — com os mesmos hiperparâmetros, alcançando **96,2% de acurácia**
e o mesmo padrão de desempenho por classe, o que valida a metodologia do estudo.

## Resultados da reprodução

| Métrica | Artigo original | Nossa reprodução |
|---|---|---|
| **Acurácia (teste)** | **97,4%** | **96,2%** |
| F1 — Negativo | 0,96 | 0,92 |
| F1 — Neutro | 0,98 | 0,98 |
| F1 — Positivo | 0,97 | 0,96 |
| Macro F1 | 0,97 | 0,95 |

Detalhes por classe (nossa execução, conjunto de teste com 894 tweets):

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| Negativo | 0,91 | 0,92 | 0,92 | 158 |
| Neutro | 0,97 | 0,98 | 0,98 | 451 |
| Positivo | 0,98 | 0,95 | 0,96 | 285 |

A diferença de ~1,2 p.p. é esperada: o artigo original não fixa semente aleatória
e não documenta a rotulação. A matriz de confusão reproduz o padrão qualitativo
do estudo (classe neutra quase perfeita, classe negativa — minoritária — mais fraca).
A curva de treinamento evidencia **sobreajuste a partir da epoch 4** (val loss sobe
de 0,092 para 0,147), indicando que as 10 epochs do artigo são excessivas.

## Decisões metodológicas (lacunas do artigo original)

1. **Rotulação (não documentada no artigo).** O dataset público não tem rótulos.
   Comparamos rotuladores léxicos com a distribuição reportada nas Figuras 9–10 do
   artigo (~17% neg / 56% neu / 28% pos): o VADER não reproduz a distribuição
   (43/26/31), mas a **polaridade do TextBlob** (`<0` → negativo, `=0` → neutro,
   `>0` → positivo) reproduz com alta fidelidade (**17,6 / 50,5 / 31,9%**).
   Adotamos o TextBlob como estratégia de rotulação.
2. **Semente aleatória.** O artigo declara não fixar seed; fixamos `seed = 42`
   (NumPy, PyTorch e particionamento) com divisão **estratificada** por classe:
   16.082 treino / 894 validação / 894 teste.
3. **Compatibilidade de bibliotecas.** O método `encode_plus`, usado em códigos
   da época do artigo, foi removido em versões recentes do `transformers`;
   a implementação usa a chamada direta do tokenizador.
4. **Precisão mista (AMP/fp16)** no treino, para reduzir o tempo (~50 min na
   GPU T4) sem alterar nenhum hiperparâmetro.

## Hiperparâmetros (idênticos ao artigo)

| Hiperparâmetro | Valor |
|---|---|
| Modelo | `bert-base-multilingual-cased` |
| Epochs | 10 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Max sequence length | 160 |
| Otimizador | AdamW (scheduler linear, sem warmup) |
| Divisão | 90% treino / 5% validação / 5% teste |

## Dados

Dataset público do Kaggle:
[Russia vs Ukraine Tweets](https://www.kaggle.com/datasets/vanamayaswanth/russia-vs-ukraine-tweets)
— baixe o `War.csv` e coloque em `data/`. São 17.951 tweets (17.870 após limpeza)
com duas colunas: data/hora e texto em formato byte-string (`b'...'`).

## Como executar

### No ambiente

1. Abra `notebooks/reproducao_bert_colab.ipynb`.
2. Ative a GPU se puder: *Ambiente de execução → Alterar tipo → T4 GPU*.
3. Execute as células em ordem (upload do `War.csv` quando solicitado).
4. Tempo estimado: **~50 min** (10 epochs na T4 com AMP).
5. Ao final, baixe os artefatos (`history.json`, `classification_report.txt`,
   `training_history.png`, `confusion_matrix.png`).

### Opção 2 — Local (requer GPU NVIDIA)

```bash
pip install -r requirements.txt
python -m src.preprocessing --input data/War.csv --output data/tweets_clean.csv
python -m src.labeling --input data/tweets_clean.csv --output data/tweets_labeled.csv
python -m src.train --data data/tweets_labeled.csv --out results/
python -m src.evaluate --model results/best_model.bin --test results/test_split.csv
```

## Saídas geradas

- `results/best_model.bin` — pesos do melhor modelo (maior acurácia de validação)
- `results/history.json` — acurácia/loss de treino e validação por epoch
- `results/classification_report.txt` — acurácia, precision, recall e F1 por classe
- `results/confusion_matrix.png` — matriz de confusão do teste
- `results/training_history.png` — curva de treino × validação

## Limitações e análise crítica

- **Circularidade da rotulação:** se os rótulos vêm de um léxico (TextBlob), a
  acurácia mede a capacidade do BERT de imitar esse léxico — tarefa mais simples
  que reconhecer sentimento humano. Os ~97% devem ser lidos com essa ressalva.
- **Sobreajuste:** 10 epochs excedem a recomendação de 2–4 do paper original do
  BERT (Devlin et al., 2019); early stopping na epoch 3–4 daria o mesmo resultado
  com cerca de 1/3 do custo computacional.
- **Desbalanceamento:** a classe negativa (17,6% do corpus) concentra os piores
  resultados (F1 0,92) — limitação apontada pelos próprios autores.

## Referências principais

- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding*. NAACL-HLT.
- Julianto, M. F.; Malau, Y.; Hidayat, W. F. (2022). *Sentiment Analysis of
  Twitter's Opinion on The Russia and Ukraine War Using Bert*. Jurnal Riset
  Informatika, 5(1).
- Sun, C. et al. (2019). *How to Fine-Tune BERT for Text Classification?* CCL.
