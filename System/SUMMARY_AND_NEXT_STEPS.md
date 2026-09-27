# Resumo Completo e Guia Passo a Passo

**Sistema de Deteção de Erros e Dificuldades de Aprendizagem — Tese de Miguel Rodrigues (MEIA, ISEP)**

Este documento resume tudo o que foi feito até agora e dá um guia detalhado do que fazer a seguir. Serve como ponto de referência único — não precisas de reconstruir o raciocínio a partir da conversa.

---

## PARTE 1 — RESUMO DO QUE FOI FEITO

### 1.1 Ponto de partida

Quando começámos, a tese tinha:
- Capítulo 1 (Introdução) e Capítulo 2 (Estado da Arte) escritos e sólidos.
- Capítulos 3–6 escritos, mas a descrever o **POC** (proof-of-concept com Sentence-BERT, pasta `Experimentation/`, 66.7% de accuracy em 42 amostras sintéticas) como se fosse o sistema final — não era.
- Nenhuma estratégia de dados concreta e verificada.
- Nenhuma implementação real do sistema descrito nos capítulos 3-6.

### 1.2 Decisões de âmbito tomadas

1. **A tese não exige português** — confirmaste que o texto da tese nunca especificou a língua. Isto foi decisivo: os melhores corpora públicos anotados de erros (BEA-2019, CoNLL-14, JFLEG) são em inglês; o equivalente em português (COPLE2) é de adultos a aprender português como língua estrangeira, um domínio diferente do que precisamos (crianças nativas com disortografia).
2. **Ordem de trabalho**: implementar primeiro, escrever os capítulos definitivos depois — evita escrever prosa "final" sobre um sistema que ainda vai mudar.
3. **Consolidar todas as fontes de dados antes de escrever código** — feito antes de qualquer implementação.
4. **O módulo de matemática não podia ser só simbólico** — apontaste corretamente que um simulador de regras não chega para uma tese de Engenharia de IA. Resolvido adicionando um classificador de Machine Learning e comparando os dois métodos.
5. **Tens uma RTX 5070 (12GB)** disponível para treino real em GPU — isto define o que é possível fazer a sério (fine-tuning de um transformer) versus o que só foi testado en miniatura neste sandbox (sem GPU).

### 1.3 Estratégia de dados final (por módulo)

**Módulo linguístico:**
| Fonte | Uso | Acesso |
|---|---|---|
| BEA-2019 (Write&Improve+LOCNESS) | Dados de treino reais | Download direto: `cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz` |
| JFLEG | Avaliação (benchmark de fluência) | GitHub `keisks/jfleg`, livre |
| CoNLL-2014/NUCLE, Cambridge FCE | Extensão futura opcional | Exigem licença assinada — não usados agora |

**Módulo matemático:**
| Fonte | Uso | Acesso |
|---|---|---|
| MaE (math-misconceptions) | 55 misconceções de álgebra validadas por professores, 220 exemplos reais | GitHub `nancyotero-projects/math-misconceptions` (o espelho no HuggingFace só tem ficheiros soltos, não os dados estruturados — usar o GitHub) |
| DeepMind `mathematics_dataset` | **Não usado** — está sem manutenção e quebra com o SymPy atual | Substituído por gerador próprio |

**Módulo de profiling:** não existe nenhum dataset público (o mesmo motivo de sempre: dados longitudinais reais de alunos menores exigem aprovação ética + consentimento parental). Resolvido com um simulador de arquétipos sintéticos.

### 1.4 O que foi implementado, testado e verificado

Tudo está em `C:\ISEP\TESE\System\` (pasta nova, separada do `Experimentation/` que é só o POC antigo).

**Matemática (`System/math/`)**
- `mal_rules.py` — simulador simbólico das 5 mal-rules do Cap.3 (SFL, borrow-from-zero, borrow-no-decrement, frac-add, frac-compare). Testado exatamente contra os exemplos já escritos no `chapter3.tex` (43-17=34, 302-158=254, 52-27=35, etc.) — bate certo com o texto que já tinhas.
- `features.py` + `train_ml_classifier.py` — classificador de Machine Learning (Gradient Boosting) treinado nos mesmos problemas, comparado diretamente com o método simbólico.
- **Resultado real (já corrido):** em respostas exatas, os dois métodos acertam 100%. Mas quando simulo um aluno que aplica a mal-rule *e* comete um deslize extra de ±1 num dígito (cenário realista), o classificador ML mantém 61.7% de exatidão enquanto o método simbólico cai para 0% (porque exige correspondência exata por construção). Isto é um resultado limpo, real e citável sobre o compromisso entre precisão e robustez — responde diretamente à tua Q3 ("symbolic methods and simple ML classifiers").

**Linguística (`System/linguistic/`)**
- `taxonomy_map.py` — mapeia os tipos de erro genéricos do ERRANT para a tua taxonomia (ERR-PHONO/ORTHO/SEG). Validado contra dados **reais** do BEA-2019 (não só frases de exemplo) — isto apanhou e corrigiu 3 bugs de precisão genuínos (maiúsculas e pontuação a "vazar" para segmentação, deteção de homófonos demasiado permissiva).
- `synthetic_corruption.py` — gerador de corrupção baseado em regras (substituição fonológica, regras ortográficas, homófonos, fusão/separação de palavras).
- `build_real_bio_corpus.py` — construiu **2584 frases reais anotadas** a partir do BEA-2019.
- `train_linguistic_model.py` — script de treino com o currículo em 3 fases (sintético → real → combinado), com um interruptor `SMOKE_TEST` (1 = teste rápido em CPU, já validado; 0 = treino real em GPU com DeBERTa-v3-base).
- **Estado:** só foi feito um "smoke test" (modelo pequeno, poucos dados, CPU, 2 minutos) para provar que o pipeline funciona sem erros. Não há ainda um modelo treinado a sério — isso precisa da tua GPU.

**Profiling (`System/profiling/`)**
- `simulate_and_cluster.py` — simula alunos sintéticos com 6 arquétipos de dificuldade diferentes, e testa se o clustering (KMeans/HDBSCAN) consegue recuperá-los a partir só dos padrões de erro.
- **Resultado real:** com parâmetros fracos (poucos dados por aluno, arquétipos pouco distintos), o KMeans só recuperava os arquétipos com ARI≈0.53 e o HDBSCAN falhava completamente. Com parâmetros mais fortes, o KMeans chega a ARI≈0.97. Achado prático: um sistema real precisa de um número mínimo de respostas observadas por aluno antes dos perfis serem fiáveis — vale a pena quantificar isto no Cap.5.

### 1.5 Descobertas que valem a pena citar na tese

Estas não são só "coisas que corrigimos" — são achados metodológicos genuínos:

1. SFL e BORROW_NO_DEC são matematicamente idênticos para subtrações de 2 dígitos com um único empréstimo — só se distinguem a partir de 3 dígitos.
2. O malrule_frac_compare concorda com a resposta certa por coincidência sempre que os denominadores são iguais — nem sempre revela o "bug".
3. O classificador ML de matemática generaliza sob ruído; o método simbólico não, por definição.
4. A recuperação de perfis de dificuldade por clustering depende fortemente da quantidade de dados observados por aluno.
5. O DeepMind `mathematics_dataset` está morto/incompatível — documentado para não perderes tempo a tentar usá-lo.
6. O dataset MaE no HuggingFace não tem os dados estruturados (só ficheiros soltos) — usar sempre o GitHub original.

### 1.6 Estado atual, em uma frase por módulo

- **Matemática:** funciona hoje, sem preparação — dá-lhe um problema e uma resposta, ele classifica corretamente (com as limitações documentadas).
- **Linguística:** o "cérebro" ainda não aprendeu nada a sério — só provámos que a receita de treino não tem erros. Precisa do treino real na tua GPU.
- **Profiling:** funciona sobre dados sintéticos; só terá utilidade real depois de a linguística estar treinada a sério.

---

## PARTE 2 — PASSO A PASSO DETALHADO (O QUE FAZER A SEGUIR)

### Passo 1 — Preparar o ambiente na tua máquina

**Atenção a dois pontos específicos da tua máquina, descobertos ao tentar isto pela primeira vez:**

1. **Usa Python 3.11 ou 3.12 para este ambiente virtual — não o 3.14.** O spaCy (usado pelo ERRANT) depende de pacotes (thinc, blis) que ainda não têm pacotes pré-compilados para Python 3.14 em meados de 2026, o que faz o pip tentar compilar do zero e falhar. Não precisas de desinstalar o 3.14 — só criar o ambiente virtual com outra versão.
2. **A tua RTX 5070 precisa da build CUDA 12.8 (`cu128`), não `cu121`.** A série RTX 50 (Blackwell, sm_120) é demasiado recente para os kernels do CUDA 12.1 — com `cu121` o PyTorch instala mas não consegue mesmo assim usar a GPU.

```bash
# 1. Instala Python 3.11 (ou 3.12) a partir de python.org, se ainda não tiveres.
# 2. Cria o ambiente virtual com essa versão específica:
py -3.11 -m venv venv
venv\Scripts\activate

# 3. Instala tudo menos o torch a partir do requirements.txt:
pip install -r requirements.txt

# 4. Instala o PyTorch com a build CUDA certa para a tua placa:
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 5. Confirma que a GPU é vista corretamente ANTES de continuares:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Tem de imprimir: True   NVIDIA GeForce RTX 5070
```

Se ainda der `False`, confirma em `pytorch.org/get-started/locally` qual é o índice CUDA recomendado atual para a tua versão de driver — pode já ter saído um `cu129` ou mais recente entretanto.

Só depois disto instala o modelo do spaCy:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
```

### Passo 2 — (Recomendado, mas opcional) Melhorar os dados sintéticos linguísticos antes de treinar a sério

Atualmente, `synthetic_corruption.py` usa só 20 frases de exemplo escritas à mão (`SEED_SENTENCES`). Antes do treino real, o ideal é substituir isto por frases reais e corretas retiradas do lado "correto" do BEA-2019 (que já tens descarregado, ou no `data/linguistic_real/` se ainda lá estiver). Isto dá muito mais variedade ao gerador sintético. Se preferires avançar já sem isto, tudo bem — é só uma melhoria, não um bloqueio.

### Passo 3 — Correr o treino real do modelo linguístico

```bash
cd System/linguistic
set SMOKE_TEST=0
python train_linguistic_model.py
```

(No PowerShell: `$env:SMOKE_TEST="0"`)

O que vai acontecer: o script carrega os 2000 exemplos sintéticos + 2584 reais, treina o `DeBERTa-v3-base` em 3 fases (só sintético → só real → combinado), com batch size 16 (ajustado para 12GB de VRAM). Deve demorar entre alguns minutos a cerca de uma hora, dependendo de quantos epochs decidires correr (já vem configurado com 3-4 epochs por fase).

**Enquanto corre**, podes acompanhar o uso da GPU noutro terminal:

```bash
nvidia-smi -l 2
```

### Passo 4 — Analisar os resultados

No fim, o script imprime as métricas de avaliação (precision/recall/F1) de cada fase. O que procurar:

1. **F1 da Fase 3 (combinado) vs. Fase 1 (só sintético) vs. Fase 2 (só real)** — isto responde diretamente à tua Hipótese H2 (o sintético + real combinado é melhor do que qualquer um sozinho?).
2. Por categoria de erro (PHONO/ORTHO/SEG) — vê se alguma categoria está claramente pior; isso é normal e digno de discussão no Cap.5 (ex: SEG pode ter menos exemplos reais no BEA-2019, como vimos: só 442 dos 42000 edits mapeados).
3. Guarda o output completo (copia o terminal para um ficheiro de texto) — vais precisar destes números exatos para escrever o Cap.5.

Manda-me os números quando tiveres — atualizamos os capítulos com resultados reais nessa altura, não antes.

### Passo 5 — Estender a comparação matemática

O `train_ml_classifier.py` só testou a comparação "limpo vs. ruidoso" para subtração. Passos naturais a seguir (posso fazer isto no sandbox, não precisa de GPU):

1. Repetir a mesma comparação limpo/ruidoso para as frações.
2. Testar o classificador ML diretamente nos 220 exemplos reais do MaE (não só no nosso conjunto sintético de teste) — isto dá uma validação externa genuína.
3. Experimentar outros classificadores simples (Random Forest, Regressão Logística) para ver se o resultado se mantém.

### Passo 6 — Reescrever os capítulos 3–6 com resultados reais

Só depois de teres os números reais do Passo 4 (e opcionalmente do Passo 5), faz sentido reescrever:

- **Cap.3**: atualizar a secção "Data Strategy" com as fontes reais (não COPLE2/ASSISTments), e a secção "Proposed Methods" para descrever os dois métodos matemáticos comparados (não só um).
- **Cap.4**: descrever a arquitetura real implementada (os três módulos tal como estão em `System/`).
- **Cap.5**: reportar os números reais de treino (linguística) e os resultados já obtidos (matemática, profiling).
- **Cap.6**: rever conclusões e limitações à luz do que realmente foi feito.

---

## Ficheiros de referência

- `System/README.md` — estado técnico atual de cada módulo.
- `System/RUN_ON_GPU.md` — instruções específicas de execução em GPU (repetidas e resumidas aqui).
- `Thesis document/data_strategy_plan.md` — plano de dados completo.
- `Thesis document/model_training_plan.md` — plano técnico original.
- `System/data/math_ml_vs_symbolic_results.json` — números exatos da comparação matemática.
- `System/data/profiling_smoke_results.json` — números exatos do profiling.
