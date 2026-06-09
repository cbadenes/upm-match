# Consultas para probar MAGE de Cypher
Están a priori optimizadas para la primera subida de datos: 500 PDI - 26K publicaciones - 4K asignaturas - 297 posiciones

## Centralidad
Identificación de qué nodos (vértices) son los más importantes, influyentes o críticos dentro de una estructura.

### PageRank de solo investigadores
Filtramos solo nodos de PDIs para evitar procesar las publicaciones. Revela quiénes tienen más influencia directa en la red.
> **Qué revela**: investigadores clave que conectan muchas publicaciones y colaboradores
```Cypher
CALL pagerank.get() YIELD node, rank
WITH node, rank
WHERE "Pdi" IN labels(node)
RETURN node.name AS investigador, rank
ORDER BY rank DESC
LIMIT 20
```

### Betweenness centrality - puentes entre comunidades
Identifica investigadores que actúan como puentes entre grupos de investigación distintos. Es costoso, tarda un rato en ejecutar, si o si hay que usar el `LIMIT`.
> **Qué revela**: un `PDI` con alta *betweenness* pero bajo *PageRank* es un conector estratégico, no un prolífico publicador.
```Cypher
CALL betweenness_centrality.get(FALSE, TRUE) YIELD node, betweennes_centrality
WITH node, betweennes_centrality
WHERE "Pdi" in labels(node)
RETURN node.name AS investigador, betweennes_centrality
ORDER BY betweennes_centrality DESC
LIMIT 20
```

### Degree centrality - los más conectados
Esta consulta es la más rápida. Muestras cuántas conexiones directas tiene cada nodo.
> **Qué revela**: compara *degree* de `PDI` vs `Publication` para detectar investigadores hiperprolíficos.
```Cypher
CALL degree_centrality.get() YIELD node, degree
WITH node, degree
WHERE "Pdi" IN labels(node)
RETURN node.name AS investigador, degree AS conexiones_totales
ORDER BY conexiones_totales DESC
LIMIT 25
```

### HITS - hubs y autoridades
Separa dos roles: hubs (colabora con muchos autores relevantes) vs autoridades (muy referenciado). Más rico que PageRank para grafos dirigidos.
> **Qué revela**: hub alto + autoridad baja = colaborador activo. Autoridad alta + hub bajo = experto aislado.
#### NO FUNCIONA: There is no procedure named 'hits.get'.
```Cypher
CALL hits.get() YIELD node, hub, authority
WITH node, hub, authority
WHERE "Pdi" IN labels(node)
RETURN node.name AS investigador,
      round(hub * 10000) / 10000 AS hub_score,
      round(authority * 10000) / 10000 AS authority_score
ORDER BY authority_score DESC
LIMIT 20
```

## Comunidades
Identificación de grupos de investigadores que colaboran intensamente entre sí. Esencial para entender la estructura organizacional emergente del grafo.

### Detección de comunidades - Louvain
Agrupa investigadores en comunidades por estructura de colaboración. Mejor balance calidad/velocidad.
> **Qué revela**: cada `community_id` es un grupo natural de investigación. Podemos usar `sample_members` para nombrarlo.
```Cypher
CALL community_detection.get()
YIELD node, community_id
WITH node, community_id
WHERE "Pdi" IN labels(node)
RETURN community_id,
      count(*) AS num_miembros,
      collect(node.name)[..5] AS muestra_miembros
ORDER BY num_miembros DESC
LIMIT 15
```

### Asignaturas por comunidad - ¿qué enseña cada grupo?
Cruza la detección de comunidades con las asignaturas para caracterizar cada grupo de investigación.
>**Qué revela**: permite ver si los grupos de investigación son cohesivos en docencia, o si hay PDI cross-disciplinar.
```Cypher
CALL community_detection.get()
YIELD node, community_id
WITH node AS pdi, community_id
WHERE "Pdi" IN labels(pdi)
MATCH (pdi)-[:Teaching]->(s:Subject)
RETURN community_id,
      count(DISTINCT s) AS asignaturas_distintas,
      collect(DISTINCT s.name)[..6] AS asignaturas_muestra
ORDER BY community_id
```

### Label propagation - alternativa más rápida
Más rápido que *Louvain*, útil para primera exploración. Menos estable entre ejecuciones.
>**Qué revela**: usado para exploración rápida; sirve para confirmar con *Louvain* si se encuentra algo interesante.
```Cypher
// La anterior era community_detection_online.set(FALSE, FALSE) pero fallaba
CALL community_detection.get()
YIELD node, community_id
WITH node, community_id
WHERE "Pdi" IN labels(node)
RETURN community_id,
       count(*) AS size,
       collect(node.name)[..4] AS ejemplos
ORDER BY size DESC
LIMIT 10
```

## Similitud
Medidas de proximidad y overlap entre investigadores. Detecta colaboraciones, aislamiento y patrones de coautoría en la red.

### Similitud Jaccard entre pares de investigadores
Mide cuántas publicaciones comparten dos investigadores (intersección / unión). Detecta colaboradores frecuentes.
>**Qué revela**: $Jaccard > 0.3$ indica colaboración sólida. Busca pares con muchas pubs compartidas pero *jaccard* bajo: colaboración puntual en grupos grandes.
```Cypher
MATCH (p1:Pdi)-[:Authorship]->(pub:Publication)
      <-[:Authorship]-(p2:Pdi)
WHERE id(p1) < id(p2)
WITH p1, p2, count(pub) AS shared
MATCH (p1)-[:Authorship]->(pub1:Publication)
WITH p1, p2, shared, count(pub1) AS total1
MATCH (p2)-[:Authorship]->(pub2:Publication)
WITH p1, p2, shared, total1, count(pub2) AS total2
RETURN p1.name, p2.name,
       round(toFloat(shared) / (total1 + total2 - shared) * 1000) / 1000 AS jaccard,
       shared AS publicaciones_compartidas
ORDER BY jaccard DESC
LIMIT 20
```

### PDI sin colaboraciones - nodos aislados
Investigadores que nunca han coautorado con otro PDI del dataset
>**Qué revela**: puede indicar investigadores que colaboran más con otras universidades o muy especializados.
```Cypher
MATCH (p:Pdi)
WHERE NOT (p)-[:Authorship]->(:Publication)
      <-[:Authorship]-(:Pdi)
OPTIONAL MATCH (p)-[:Authorship]->(pub:Publication)
RETURN p.name AS investigador,
       count(pub) AS publicaciones_solo
ORDER BY publicaciones_solo DESC
LIMIT 20
```

## Estructura
Análisis topológico del grafo: conectividad, componentes, densidad y propiedades globales. Fundamental para evaluar la calidad e integridad de los datos.

### Componentes débilmente conexas rápido
¿Está todo el grafo conectado o hay islas? Revela subgrafos completamente separados.
>**Qué revela**: Si hay muchas componentes pequeñas puede haber datos sucios o PDI sin conexión con el grupo principal.
#### Devuelve resultado vacío
```Cypher
CALL weakly_connected_components.get()
YIELD node, component_id
WITH node, component_id
WHERE "Pdi" IN labels(node)
RETURN component_id,
       count(*) AS tam,
       collect(node.name)[..3] AS miembros
ORDER BY tam DESC
LIMIT 15
```

### Estadísticas generales del grafo rápido
Resumen rápido de la topología: densidad, grado medio, número de componentes.
> **Qué revela**: Baseline necesario antes de interpretar cualquier otro algoritmo.
#### Tarda mucho en ejecutar
```Cypher
CALL graph_analyzer.analyze()
YIELD name, value
RETURN name, value
ORDER BY name
```

## Caminos
Algoritmos de búsqueda y distancias en el grafo. Calcula caminos entre nodos, proximidad y el "número de Erdős académico" de la red.

### Distancia entre investigadores - BFS rápido
¿Cuántos pasos separan a dos investigadores a través de coautorías? El número de Erdős académico.
> **Qué revela**: Si la mayoría están a ≤3 saltos, la red es densa y bien conectada.
#### NO FUNCIONA: There is no procedure named 'bfs.get'.
```Cypher
// Hay que sustituir los nombres por los que se quieran comparar
MATCH (inicio:Pdi {name: "Nombre PDI A"}), (fin:Pdi {name: "Nombre PDI B"})
CALL bfs.get(inicio, fin, "Authorship")
YIELD path
RETURN [n IN nodes(path) |
      coalesce(n.name, n.title)] AS camino,
      length(path) AS distancia
```

### Distancia media desde una muestra de PDI lento
Calcula la distancia media entre un subconjunto de `PDI` y todos los demás. Equivale a *closeness centrality* con control de coste.
> **Qué revela**: PDI con distancia media baja son los más centrales y accesibles de la red.
```Cypher
// 20 PDI aleatorios como origen
MATCH (p:Pdi)
WITH p ORDER BY rand() LIMIT 20
MATCH path = (p)-[:Authorship *1..4]->
              (:Publication)<-[:Authorship *1..4]-
              (otros:Pdi)
WHERE p <> otros
RETURN p.name AS origen,
       avg(length(path)) AS distancia_media,
       count(DISTINCT otros) AS alcanzados
ORDER BY distancia_media
LIMIT 20
```

## Colaboración y Recomendaciones

### Recomendación de colaboradores - basada en amigos comunes
Identifica investigadores que aún no colaboran pero que comparten muchos coautores. Métrica para predecir futuras colaboraciones.
> **Qué revela**: p1 y p2 no han coautorado pero tienen N amigos comunes $\rightarrow$ candidatos para futuras colaboraciones. Filtrar por `amigos_comunes >= 3` para oportunidades estratégicas.
```Cypher
MATCH (p1:Pdi)-[:Authorship]->(:Publication)<-[:Authorship]-(mid:Pdi)
MATCH (p2:Pdi)-[:Authorship]->(:Publication)<-[:Authorship]-(mid)
WHERE id(p1) < id(p2) AND NOT (p1)-[:Authorship]->(:Publication)<-[:Authorship]-(p2)
WITH p1, p2, count(DISTINCT mid) AS amigos_comunes
WHERE amigos_comunes >= 3
RETURN p1.name AS investigador_1,
       p2.name AS investigador_2,
       amigos_comunes
ORDER BY amigos_comunes DESC
LIMIT 20
```

### Clustering coefficient local - cohesión de subgrupos
Mide cuánto conectados están los colaboradores de un PDI entre sí. Alto = clúster muy cohesionado. Bajo = PDI conector entre grupos.
> **Qué revela**: PDI con clustering alto pertenecen a núcleos cerrados. PDI con clustering bajo son puentes o conectores entre comunidades.
```Cypher
MATCH (p:Pdi)-[:Authorship]->(:Publication)<-[:Authorship]-(c1:Pdi)
MATCH (p)-[:Authorship]->(:Publication)<-[:Authorship]-(c2:Pdi)
WHERE id(c1) < id(c2)
WITH p, c1, c2
OPTIONAL MATCH (c1)-[:Authorship]->(:Publication)<-[:Authorship]-(c2)
WITH p, c1, c2, CASE WHEN c1 IS NOT NULL AND c2 IS NOT NULL THEN 1 ELSE 0 END AS pair_exists
WITH p, count(*) AS total_pares, sum(pair_exists) AS conexiones
WHERE total_pares > 0
WITH p, toFloat(conexiones) / total_pares AS clustering_coef
RETURN p.name AS investigador, clustering_coef
ORDER BY clustering_coef DESC
LIMIT 25
```

### Asignaturas e investigadores - caracterización por docencia
Relaciona los temas que enseña cada investigador con sus colaboraciones. Identifica si las colaboraciones son cross-disciplinares.
> **Qué revela**: Si dos coautores enseñan asignaturas muy diferentes $\rightarrow$ colaboración multidisciplinar. Si las asignaturas solapan $\rightarrow$ grupo temático cohesionado.
```Cypher
MATCH (p1:Pdi)-[:Authorship]->(pub:Publication)<-[:Authorship]-(p2:Pdi),
      (p1)-[:Teaching]->(s1:Subject),
      (p2)-[:Teaching]->(s2:Subject)
WHERE id(p1) < id(p2)
WITH p1.name AS pdi1, p2.name AS pdi2,
     count(pub) AS pubs_comunes,
     count(DISTINCT s1) AS asignaturas_pdi1,
     count(DISTINCT s2) AS asignaturas_pdi2,
     count(DISTINCT s1.name + ", " + s2.name) AS solapamiento
RETURN pdi1, pdi2, pubs_comunes,
       asignaturas_pdi1, asignaturas_pdi2,
       solapamiento
ORDER BY pubs_comunes DESC
LIMIT 20
```

## Productividad y Actividad

### Ranking de productividad - PDI más prolíficos
Clasificación por número de publicaciones, coautores únicos y centralidad combinada. Proxy de impacto académico.
> **Qué revela**: Compara *cantidad* (pubs) vs *amplitud* (coautores) vs *centralidad* (influencia). Identifica investigadores prolíficos pero aislados.
```Cypher
MATCH (p:Pdi)-[:Authorship]->(pub:Publication)
WITH p, count(DISTINCT pub) AS num_publicaciones
MATCH (p)-[:Authorship]->(:Publication)<-[:Authorship]-(coautor:Pdi)
WITH p, num_publicaciones, count(DISTINCT coautor) AS num_coautores
OPTIONAL MATCH (p)-[:Teaching]->(:Subject)
WITH p, num_publicaciones, num_coautores, count(DISTINCT 1) AS asignaturas
RETURN p.name AS investigador,
       num_publicaciones,
       num_coautores,
       asignaturas,
       round(toFloat(num_publicaciones) / (num_coautores + 1) * 10) / 10 AS pubs_por_coautor
ORDER BY num_publicaciones DESC
LIMIT 30
```

### Grupos de investigación informales - cliques y casi-cliques
Identifica grupos en los que casi todos colaboran con casi todos ($densidad > 80%$). Son posibles grupos de investigación informales.
> **Qué revela**: Grupos altamente cohesionados. Si no tienen líder claro (todos tienen *similar degree*), son más resilientes.
```Cypher
MATCH (p:Pdi)-[:Authorship]->(:Publication)<-[:Authorship]-(v:Pdi)
WITH p, collect(DISTINCT v) AS vecinos
WHERE size(vecinos) >= 3
UNWIND vecinos AS v1
UNWIND vecinos AS v2
WITH p, vecinos, v1, v2
WHERE id(v1) < id(v2)
OPTIONAL MATCH (v1)-[:Authorship]->(:Publication)<-[:Authorship]-(v2)
WITH p, vecinos, v1, v2, count(*) AS conexiones
WITH p, vecinos,
     count(*) AS posibles,
     sum(CASE WHEN conexiones > 0 THEN 1 ELSE 0 END) AS conexiones_validas
WHERE posibles > 0
WITH p, size(vecinos) AS tam_grupo, toFloat(conexiones_validas) / posibles AS densidad
WHERE tam_grupo >= 4 AND densidad > 0.5
RETURN p.name AS investigador_central,
       tam_grupo,
       round(densidad * 1000) / 1000 AS densidad_grupo
ORDER BY densidad DESC, tam_grupo DESC
LIMIT 20
```
<!-- Esta otra es lo mismo pero con una lista de los colaboradores, aunque no estoy seguro de si es igual, no devuelve los mismo resultados
```Cypher
// 1. Identificar investigadores y sus vecinos únicos
MATCH (p:Pdi)-[:Authorship]->(:Publication)<-[:Authorship]-(v:Pdi)
WHERE p <> v
WITH p, collect(DISTINCT v) AS vecinos
WHERE size(vecinos) >= 3

// 2. Desplegar los pares para evaluar conexiones
UNWIND vecinos AS v1
UNWIND vecinos AS v2
WITH p, vecinos, v1, v2
WHERE id(v1) < id(v2)

// 3. Evaluar conexiones y mantener 'vecinos' en el flujo
OPTIONAL MATCH (v1)-[:Authorship]->(pub:Publication)<-[:Authorship]-(v2)
WITH p, vecinos, 
     count(v1) AS total_pares, 
     count(pub) AS conexiones_reales

// 4. Filtrar por densidad (manteniendo la variable 'vecinos')
WITH p, vecinos, 
     size(vecinos) AS tam_grupo,
     toFloat(conexiones_reales) / total_pares AS densidad
WHERE tam_grupo >= 4 AND densidad > 0.5

// 5. Retornar el investigador, el tamaño, la lista y la densidad
RETURN p.name AS investigador_central, 
       tam_grupo, 
       [n in vecinos | n.name] AS lista_nombres_vecinos, 
       densidad
ORDER BY densidad DESC, tam_grupo DESC
LIMIT 20
``` -->
