## Comparativa por modo de búsqueda

Modos: `lexical_direct` = Léxica directa (control) · `keywords` = Léxica con LLM · `summary_direct` = Semántica directa · `summary_enriched` = Semántica enriquecida · `hybrid` = Híbrida · `rerank_listwise` = Reordenación listwise (texto) · `rerank_listwise_kg` = Reordenación listwise (grafo+texto) · `clarify_hybrid` = Híbrida con clarificación

| mode | n | mrr | map | hit@1 | hit@5 | hit@10 | hit@25 | ndcg@5 | ndcg@10 |
|---|---|---|---|---|---|---|---|---|---|
| clarify_hybrid | 481 | 0.153 | 0.153 | 0.083 | 0.220 | 0.295 | 0.414 | 0.153 | 0.177 |
| hybrid | 481 | 0.150 | 0.150 | 0.079 | 0.214 | 0.297 | 0.420 | 0.148 | 0.175 |

## Tiempo por consulta (segundos)

| mode | n_timed | total_mean_s | total_median_s | total_p90_s | llm_mean_s | search_mean_s |
|---|---|---|---|---|---|---|
| hybrid | 481 | 19.659 | 19.501 | 21.703 | 13.339 | 6.320 |
| clarify_hybrid | 481 | 53.865 | 63.804 | 72.698 | 39.514 | 14.351 |

## hybrid — desglose por study_type

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| study_type=TFG | 238 | 0.138 | 0.063 | 0.210 | 0.298 | 0.167 |
| study_type=TFM | 243 | 0.161 | 0.095 | 0.218 | 0.296 | 0.182 |

## hybrid — desglose por knowledge_area

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| knowledge_area=ETSII | 154 | 0.106 | 0.045 | 0.149 | 0.240 | 0.128 |
| knowledge_area=ETSISI | 105 | 0.148 | 0.086 | 0.219 | 0.248 | 0.163 |
| knowledge_area=ETSIST | 73 | 0.141 | 0.055 | 0.219 | 0.301 | 0.166 |
| knowledge_area=ETSIT | 149 | 0.200 | 0.121 | 0.275 | 0.389 | 0.235 |

## hybrid — desglose por year

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| year=2004 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| year=2014 | 2 | 0.026 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2015 | 1 | 0.100 | 0.000 | 0.000 | 1.000 | 0.289 |
| year=2016 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2017 | 1 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2018 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2019 | 14 | 0.194 | 0.143 | 0.214 | 0.357 | 0.223 |
| year=2020 | 49 | 0.138 | 0.061 | 0.184 | 0.265 | 0.153 |
| year=2021 | 40 | 0.153 | 0.100 | 0.200 | 0.250 | 0.168 |
| year=2022 | 70 | 0.138 | 0.071 | 0.200 | 0.314 | 0.168 |
| year=2023 | 55 | 0.176 | 0.091 | 0.255 | 0.364 | 0.214 |
| year=2024 | 89 | 0.200 | 0.112 | 0.303 | 0.348 | 0.227 |
| year=2025 | 128 | 0.123 | 0.055 | 0.180 | 0.273 | 0.148 |
| year=2026 | 28 | 0.087 | 0.036 | 0.143 | 0.179 | 0.099 |

## clarify_hybrid — desglose por study_type

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| study_type=TFG | 238 | 0.155 | 0.080 | 0.239 | 0.311 | 0.184 |
| study_type=TFM | 243 | 0.151 | 0.086 | 0.202 | 0.280 | 0.170 |

## clarify_hybrid — desglose por knowledge_area

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| knowledge_area=ETSII | 154 | 0.113 | 0.058 | 0.162 | 0.227 | 0.131 |
| knowledge_area=ETSISI | 105 | 0.165 | 0.105 | 0.238 | 0.286 | 0.185 |
| knowledge_area=ETSIST | 73 | 0.137 | 0.055 | 0.205 | 0.315 | 0.169 |
| knowledge_area=ETSIT | 149 | 0.193 | 0.107 | 0.275 | 0.362 | 0.224 |

## clarify_hybrid — desglose por year

| segment | n | mrr | hit@1 | hit@5 | hit@10 | ndcg@10 |
|---|---|---|---|---|---|---|
| year=2004 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| year=2014 | 2 | 0.019 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2015 | 1 | 0.143 | 0.000 | 0.000 | 1.000 | 0.333 |
| year=2016 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2017 | 1 | 0.014 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2018 | 1 | 0.015 | 0.000 | 0.000 | 0.000 | 0.000 |
| year=2019 | 14 | 0.207 | 0.143 | 0.214 | 0.429 | 0.253 |
| year=2020 | 49 | 0.130 | 0.061 | 0.184 | 0.265 | 0.149 |
| year=2021 | 40 | 0.109 | 0.050 | 0.150 | 0.200 | 0.119 |
| year=2022 | 70 | 0.151 | 0.071 | 0.243 | 0.314 | 0.179 |
| year=2023 | 55 | 0.209 | 0.127 | 0.309 | 0.364 | 0.240 |
| year=2024 | 89 | 0.194 | 0.112 | 0.292 | 0.348 | 0.222 |
| year=2025 | 128 | 0.131 | 0.078 | 0.172 | 0.258 | 0.150 |
| year=2026 | 28 | 0.091 | 0.000 | 0.179 | 0.250 | 0.122 |