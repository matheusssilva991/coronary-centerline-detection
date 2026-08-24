# Pipeline híbrido de resolução

Destino dos resultados compactos de
`src/experiments/hybrid_resolution_pipeline.py`, que localiza a aorta e os
óstios em resolução média e transfere as coordenadas para segmentar as artérias
em alta resolução.

Cada execução pode conter configuração, resultados por imagem e comparação
pareada entre variantes. Máscaras intermediárias e caches não devem ser
mantidos aqui. A pasta pode ficar vazia quando não houver um experimento ativo
preservado.
