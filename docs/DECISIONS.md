# Decisões do projeto

Cada decisão relevante entra aqui: contexto, decisão, motivo e alternativas descartadas.

## 1. A causa é injetada por multiplicadores sobre um ruído pré-sorteado

- **Contexto:** o gabarito só vale se for possível provar que a causa está apenas na fatia certa.
- **Decisão:** todo o ruído é sorteado antes e sem olhar o cenário. A causa multiplica sessões, taxa de conversão e preço só nas linhas da fatia e do período.
- **Motivo:** com a mesma semente, tudo fora da fatia fica idêntico ao controle sem causa. O `test_generator` verifica isso com igualdade exata.
- **Descartado:** `Binomial` sorteado depois do efeito (muda o fluxo do gerador e quebra a igualdade fora da fatia). A variância binomial é emulada com `z × sqrt(esperado × (1 − p))`.

## 2. O gabarito fica fora do banco

- **Decisão:** `events` e `changes` vão para o SQLite; o cenário injetado vai para `ground_truth.json`, ao lado do banco.
- **Motivo:** o agente terá `run_sql`. Se o gabarito estivesse no banco, ele poderia lê-lo.

## 3. Cenário declara `changes` explicitamente

- **Decisão:** o que aparece na tabela `changes` é uma lista no próprio cenário (pode ser vazia, ou conter uma isca).
- **Motivo:** permite os casos "sem registro" (s13, s19, s20), "isca" (s03) e "mudança irrelevante" (s17) sem lógica escondida no gerador.

## 4. Magnitudes dos cenários são pontos de partida

- **Contexto:** os efeitos foram dimensionados à mão para que cada anomalia seja detectável sobre o ruído (quedas de ~5 a 29% nas métricas afetadas).
- **Decisão:** reajustar com o detector real e registrar aqui cada mudança.
- **Mudança (etapa 2):** `BASE_DAILY_SESSIONS` subiu de 50 mil para 400 mil por dia. Com 50 mil, o ruído binomial das conversões (~2,5% ao dia) escondia quedas de 8–10%: o recall caiu para 60–90% nos cenários de degrau. Continua sendo um parâmetro de projeto (um negócio maior), não um ajuste de limiar.
- **Risco conhecido:** s10 (laptop) mexe pouco na taxa de conversão agregada; o sinal está na receita e no ticket médio. Por isso a métrica-raiz de s10 alterna entre `conversion_rate` e `avg_ticket`, e o drill-down (etapa 3) precisa funcionar para as duas.

## 5. Detector: baseline de 8 semanas, não 4

- **Contexto:** a especificação original usava 4 semanas. Uma deriva lenta (s20, 28 dias) contamina uma baseline curta, porque a mediana de 4 pontos já tem 2 deles derivados.
- **Decisão:** `BASELINE_WEEKS=8`. Com 8, a mediana de 8 mesmos-dias-da-semana tem a maioria dos pontos fora da deriva.
- **Custo:** o primeiro dia com esperado é o 57º; sobram ~28 dias de resíduos históricos antes da janela de 7 dias.
- **Alternativas descartadas:** congelar a baseline em uma data fixa (exigiria saber onde começa a anomalia, que é justamente o que se quer descobrir).

## 6. Detector: resíduo relativo e escala refinada

- **Contexto:** a escala só com MAD, em ~28 pontos, oscila muito (eficiência de ~37%). Nas sementes 1000–1499 apareceram falsos alarmes raros, concentrados em `revenue` (picos de 5 a 6,5 desvios no teste de mudança sustentada).
- **Decisão:** a escala é o desvio-padrão dos resíduos limitados a ±3 escalas MAD (`CLIP_SCALES`). Mantém a robustez a pontos fora da curva e tem cauda bem mais leve: o pico observado em 1000 execuções caiu de 5,05 para 3,12 (regra de dias seguidos) e de 4,97 para 4,35 (regra sustentada).
- Os resíduos são **relativos** (`observado / esperado − 1`), porque o ruído é multiplicativo e fins de semana têm valores menores.
- **Alternativas descartadas:** subir os limiares para 4,0 e 6,0 (funcionava na calibração, mas falhou na semente 19 dos testes e custava recall); usar MAD puro.

## 7. Detector: regra de mudança sustentada

- **Contexto:** a regra da especificação (2 dias seguidos com |z| ≥ 3,5) não vê um deslocamento pequeno e persistente. s20 (deriva) ficava em 0/30.
- **Decisão:** segunda regra: `|média de z na janela| × √dias ≥ SUSTAINED_Z_THRESHOLD` (5,5), com todos os dias menos 2 no mesmo sinal.
- **Calibração:** limiares escolhidos nas sementes **1000–1499**. As sementes de teste são **0–199**. A primeira escolha (3,8 e 6,0), feita em apenas 200 sementes, falhou na semente 19 do teste; a causa foi a escala (decisão 6), não o limiar, e foi o que motivou o refinamento.
- **Resultado de calibração:** 0 falsos alarmes em 1200 execuções (s15–s18 × 300 sementes) com 3,5 e 5,0; ficou 5,5 para ter margem.
- **Limite conhecido:** s20 com fator 0,4 é detectado em ~60% das sementes. É um resultado honesto do benchmark, não um defeito escondido.

## 8. Métrica `avg_ticket` e métrica-raiz

- **Decisão:** `avg_ticket = revenue / conversions` entra como quinta métrica. A raiz é a primeira que disparou na ordem `sessions → conversion_rate → avg_ticket → conversions → revenue` (a árvore `revenue = sessions × taxa × ticket`).
- **Motivo:** mudança de preço (s10, s11) mexe no ticket antes de mexer na receita; sem `avg_ticket` o alarme viria só de `revenue`, uma métrica derivada.
- **Consequência:** o drill-down da etapa 3 precisa tratar `avg_ticket` como razão (receita por conversão), como já trata `conversion_rate`.

## 9. Drill-down: efeito de mistura nas razões

- **Contexto:** o impacto de uma fatia na taxa de conversão só pelo efeito da taxa, `(taxa_obs − taxa_esp) × w_obs`, ignora a mistura. Quando um canal perde sessões (s08, s09), a taxa total muda sem que a taxa de nenhuma fatia mude, e o drill-down não atribuiria nada.
- **Decisão:** `impacto = efeito_taxa + efeito_mistura`, com `efeito_mistura = (w_obs − w_esp) × (taxa_esp_fatia − taxa_esp_total)`. A soma dos impactos das fatias é exatamente a variação da razão (propriedade testada para as 5 métricas). Vale também para `avg_ticket` (receita por conversão).
- **Alternativa descartada:** mistura sem subtrair a taxa total. Soma igual, mas atribui impacto a uma fatia só por ganhar peso, mesmo que a taxa dela seja a média.

## 10. Drill-down: o z de uma fatia é o z da média da janela

- **Contexto:** a especificação dizia "fatia estável se `|z| < 2`". Com o z diário médio, uma janela de 2 dias tem desvio de 0,7 e uma de 7 dias tem 0,38; o mesmo limiar significava coisas diferentes. Num caso (s04, semente 1013) uma fatia com `z = −2,05` por puro ruído derrubou a causa certa e o fallback de duas causas inventou uma segunda.
- **Decisão:** `z = média dos z diários × √dias`, que é ≈ N(0, 1) em qualquer janela. `Z_STABLE = 3,5`. Medido em dados só com ruído (s15, 30 sementes × 5 dimensões): desvio-padrão do z entre 0,83 e 1,06; nenhuma fatia acima de 3,5 em janelas de 3 dias (uma de `avg_ticket` ficou em 3,9 e outra em 4,2 em janelas de 3 e 7 dias).

## 11. Drill-down: guardas contra causa inventada

- **Variação total insignificante:** com `|z| < MIN_TOTAL_Z` (3,5) a lista de causas é vazia. O EP divide por uma variação que é só ruído, então qualquer fatia parecia uma causa (em s15–s18, 100% das execuções).
- **Segunda causa:** só se procura quando o que sobra, depois de tirar a primeira, ainda tem variação significativa.
- **Materialidade (`EP_MATERIAL = 0,1`):** uma fatia irmã só desqualifica a hipótese se for instável **e** explicar pelo menos 10% da variação. Sem isso, um desvio significativo mas minúsculo numa fatia irmã (z de −4,8, EP de 0,09 em s11, semente 1036) derrubava a causa certa e abria caminho para uma segunda causa falsa.
- **Resultado (sementes 1000–1039, 20 cenários, métrica-raiz do detector):** causa certa ou, nos cenários sem causa localizável, lista vazia em 799 de 800 execuções. A exceção é uma deriva (s20) que o detector não detectou, e nesse caso o drill-down corretamente não afirma nada.

## 12. Menu: parâmetros e erros

- **Decisão:** a janela é `window_start`/`window_end`, e a baseline é `baseline_weeks` (a especificação tinha `window` e `baseline` genéricos). Pedido inválido (métrica desconhecida, janela fora dos dados, campo extra) vira um `Finding` com a chave `error`; o agente continua.
- **`check_recent_changes`:** uma mudança só conta para um `scope` se o escopo dela for global ou tiver dimensão em comum com o mesmo valor. Antes, o deploy do Sul aparecia como evidência para a hipótese `product = phone`.
- **Adiados para a etapa 6:** `run_sql` e `find_similar_incidents`.

## 13. Plano determinístico

- **Decisão:** `investigation/plan.py` roda `compare_periods`, `decompose_metric`, a busca de causas e, para cada causa, `test_hypothesis` e `check_recent_changes`. Cada passo vira `Finding`, e `Hypothesis.finding_ids` aponta para as provas. É a linha de base da ablação com o modo `llm`.
