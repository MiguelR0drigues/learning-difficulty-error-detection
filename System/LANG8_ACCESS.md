# Acesso ao Lang-8 / cLang-8 (dados reais em escala para o módulo linguístico)

## O que é

- **Lang-8 Learner Corpus (NAIST)**: o maior corpus de aprendizes de línguas gerado a partir do site Lang-8, com revisões reais feitas por falantes nativos. Inclui inglês, alemão, russo, japonês, etc.
- **cLang-8**: versão "limpa" do Lang-8, produzida pela Google Research — o lado inglês tem **2.372.119 pares (frase original → frase corrigida)**, gerados por um modelo GEC (gT5) sobre as frases-fonte do Lang-8 original.

Ambos exigem o mesmo registo — o cLang-8 não é um atalho, só reaproveita as frases-fonte do Lang-8 e junta as suas próprias correções.

## Passos de acesso (não automatizável — requer ação manual do Miguel)

1. Preencher este formulário Google: https://docs.google.com/forms/d/17gZZsC_rnaACMXmPiab3kjqBEtRHPMz0UG9Dk-x_F0k/viewform?edit_requested=true
2. Aguardar o email de resposta (não é instantâneo — é aprovação manual pela NAIST) com um link para descarregar "the raw format containing all the data up to 2010".
3. Extrair o ficheiro zip.
4. Uso restrito a fins de investigação/educação — licença **CC BY-NC-SA 4.0** (compatível com uma tese académica, mas não redistribuível comercialmente).

Página oficial: https://sites.google.com/site/naistlang8corpora

## Como isto entra no pipeline já construído

O `build_real_bio_corpus.py` atual espera dados no formato M2 (o formato de edições do BEA-2019). O Lang-8/cLang-8 vem como pares de frases (original, corrigida), não M2 — mas o ERRANT (já uma dependência do projeto, usado em `taxonomy_map.py`) sabe gerar edições M2 a partir de pares de frases diretamente via `errant_parallel`:

```bash
errant_parallel -orig source.txt -cor target.txt -out lang8_edits.m2
```

Depois disso, o pipeline existente (`m2_parser.py` → `taxonomy_map.py` → `build_real_bio_corpus.py`) já sabe processar esse ficheiro M2 exatamente como faz com o BEA-2019 — não é preciso escrever um parser novo, só apontar para o ficheiro gerado.

## Próximo passo (quando tiveres acesso)

1. Seguir os passos acima e obter `lang8_en_source.txt` / `lang8_en_target.txt` (ou equivalente do cLang-8).
2. Correr `errant_parallel` para gerar o M2.
3. Reutilizar `build_real_bio_corpus.py` apontando para esse M2 em vez do BEA-2019.
4. Juntar ao dataset real já existente (`bea2019_bio_real.json`) para uma escala muito maior de dados reais (potencialmente ~1M+ frases, vs. as 2584 atuais).

Isto fica pendente de tu preencheres o formulário — não há forma de contornar a aprovação manual.
