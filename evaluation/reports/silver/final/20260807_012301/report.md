## Comparativa por modo de búsqueda

Modos: `lexical_direct` = Léxica directa (control) · `keywords` = Léxica con LLM · `summary_direct` = Semántica directa · `summary_enriched` = Semántica enriquecida · `hybrid` = Híbrida · `rerank_listwise` = Reordenación listwise (texto) · `rerank_listwise_kg` = Reordenación listwise (grafo+texto) · `rerank_listwise_kg_excl` = Reordenación listwise (grafo+texto, exclusión) · `clarify_hybrid` = Híbrida con clarificación

| mode | n | mrr | map | hit@1 | hit@5 | hit@10 | hit@25 | ndcg@5 | ndcg@10 |
|---|---|---|---|---|---|---|---|---|---|
| rerank_listwise_kg_excl | 481 | 0.176 | 0.176 | 0.102 | 0.258 | 0.316 | 0.443 | 0.182 | 0.201 |

## Tiempo por consulta (segundos)

| mode | n_timed | total_mean_s | total_median_s | total_p90_s | llm_mean_s | search_mean_s |
|---|---|---|---|---|---|---|
| rerank_listwise_kg_excl | 481 | 25.394 | 25.476 | 27.296 | 25.394 | 0.000 |

## rerank_listwise_kg_excl — desglose por study_type

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| study_type=TFG | 238 | 0.188 | 0.109 | 0.273 | 0.315 | 0.211 |
| study_type=TFM | 243 | 0.165 | 0.095 | 0.243 | 0.317 | 0.191 |

## rerank_listwise_kg_excl — desglose por knowledge_area

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| knowledge_area=ETSII | 154 | 0.142 | 0.078 | 0.221 | 0.260 | 0.162 |
| knowledge_area=ETSISI | 105 | 0.144 | 0.086 | 0.181 | 0.229 | 0.153 |
| knowledge_area=ETSIST | 73 | 0.146 | 0.055 | 0.219 | 0.342 | 0.181 |
| knowledge_area=ETSIT | 149 | 0.250 | 0.161 | 0.369 | 0.423 | 0.283 |

## rerank_listwise_kg_excl — desglose por year

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| year=2004 | 2 | 0.167 | 0.000 | 0.500 | 0.500 | 0.250 |
| year=2014 | 2 | 0.038 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2015 | 1 | 0.500 | 0.000 | 1.000 | 1.000 | 0.631 |
| year=2016 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2017 | 1 | 0.019 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2018 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2019 | 14 | 0.164 | 0.071 | 0.286 | 0.357 | 0.203 |
| year=2020 | 49 | 0.179 | 0.082 | 0.265 | 0.286 | 0.193 |
| year=2021 | 40 | 0.182 | 0.125 | 0.225 | 0.300 | 0.202 |
| year=2022 | 70 | 0.167 | 0.086 | 0.329 | 0.371 | 0.207 |
| year=2023 | 55 | 0.189 | 0.109 | 0.255 | 0.345 | 0.219 |
| year=2024 | 89 | 0.202 | 0.124 | 0.292 | 0.360 | 0.231 |
| year=2025 | 128 | 0.176 | 0.117 | 0.227 | 0.289 | 0.193 |
| year=2026 | 28 | 0.103 | 0.036 | 0.143 | 0.179 | 0.109 |