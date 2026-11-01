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

## 14. Duas causas independentes se provam juntas (`test_hypothesis` com `together_with`)

- **Contexto:** o verificador da etapa 4 exige que toda causa do relatório tenha um `test_hypothesis` aceito. Em s12 (duas causas), cada causa sozinha explica ~53% da variação, abaixo do limiar de causa única (60%), então o plano da etapa 3 registrava como prova dois `test_hypothesis` com `accepted = false`. O relatório citava provas que o próprio código reprovava.
- **Decisão:** `test_hypothesis` ganha o parâmetro opcional `together_with` (a outra causa). Com ele, a hipótese é "as duas juntas": aceita se a união explica pelo menos `EP_MULTI_TOTAL`, as outras fatias estão estáveis e cada causa explica pelo menos `EP_MULTI_EACH` sozinha (a mesma regra de `_two_causes`). O resultado traz `together_with` e `members` (o que cada causa explica sozinha). O plano roda esse teste uma vez e as duas hipóteses apontam para ele.
- **Motivo:** a prova passa a ser a mesma regra que achou as causas, e o verificador pode ser estrito (`accepted` e mesmas dimensões).
- **Descartado:** relaxar o verificador para aceitar causas do plano sem teste aceito (tiraria a garantia "só o código confirma"); baixar o limiar de causa única (aceitaria falsas causas).

## 15. Placeholders: o código escreve todo número

- **Decisão:** o texto do LLM cita valores como `{f003.explained_share}`. O caminho segue o `result` do `Finding` (`{f004.inside.change_pct}`, `{f005.changes.0.description}`); `params.` lê os parâmetros da consulta. Só valores simples são aceitos. O código formata em pt-BR pelo nome do campo: `change_pct` com sinal e uma casa, `*_share` em porcentagem inteira, `z` com uma casa, e `observed`/`expected`/`change` de `conversion_rate` como porcentagem (a taxa é uma fração).
- **Motivo:** o LLM nunca escreve número, então não há como errar uma casa decimal nem inventar um valor. Dígito fora de placeholder é recusado, exceto versões conhecidas (`3.2`), que são valores de dimensão.
- **Limite conhecido:** número por extenso ("três causas") e negação ("não caiu") não são vistos pelo verificador.

## 16. Verificador: o que reprova

- **Decisão:** sete códigos de problema (`loose_number`, `bad_placeholder`, `unknown_finding`, `foreign_dimension`, `incoherent_status`, `unconfirmed_cause`, `contradiction`). O verificador devolve todos os problemas, não só o primeiro, para o benchmark medir a taxa de rejeição.
- **Fatos** só podem citar dimensões e valores das causas confirmadas; **opiniões** podem nomear hipóteses não confirmadas, mas qualquer número delas também vem de placeholder.
- **Direção:** verbos ("subiu", "caiu", "aumentou", "reduziu") e "queda" são comparados com o `direction` dos fatos citados (ou de todos, se nada é citado). Se os fatos têm direções mistas e o texto não cita nenhum, não há veredito. Substantivos ambíguos ("aumento de preço", "alta confiança") ficaram de fora de propósito, para não reprovar texto correto.
- **Status:** `identified` exige 1 ou 2 causas; `inconclusive` não pode ter causa nem ação proposta; fato é obrigatório.

## 17. Analista: o LLM não decide o que o código sabe

- **Decisão:** a resposta do LLM tem só `facts` e `opinions` (JSON Schema com `finding_id` restrito aos ids existentes). Status, causas e evidências são montados pelo código. Se o LLM falhar, responder fora do schema ou for reprovado pelo verificador, sai o relatório do narrador de template, sem opinião; o motivo fica em `ReportOutcome` (`rejection` ou `error`).
- **Prompt:** curto, sem resposta de exemplo; cada `Finding` aparece numa linha com os campos citáveis já formatados. Findings com erro não aparecem.
- **Narrador de template** (`agent/narrator.py`): fatos com placeholders, sem opinião. Duas causas provadas juntas viram um único fato (a estatística é da união) e as mudanças registradas são rotuladas por fatia. O relatório do narrador passa pelo verificador; se reprovar, é bug e o código levanta erro.
- **Adiados:** a ação proposta (`actions_catalog.py`, etapa 8) e o cliente Ollama (`llm/ollama.py`, etapa 6). Opinião sobre hipótese não confirmada não precisa de marca obrigatória no texto; o rótulo "não verificada" é da camada inteira.

## 18. Nada sensível sai: segredos, erros, logs e texto do banco

- **Contexto:** o projeto é público. Um segredo ou um caminho pessoal em log, traceback ou mensagem de erro vira vazamento. O Typer, por padrão, imprime as variáveis locais de cada frame nos tracebacks, o que poderia expor o DSN e o token. O `httpx` registra a URL de cada requisição, e a API do Telegram leva o token na URL.
- **Decisão:**
  - `MEMORY_DSN` e `TELEGRAM_BOT_TOKEN` são `SecretStr` (não aparecem em `repr`, `str` nem `model_dump`). `Settings.secret_values()` lista o que deve ser redigido, inclusive a senha dentro do DSN.
  - `security.redact` troca por `[REDIGIDO]` os segredos registrados, senhas em URLs de conexão, tokens do Telegram, chaves conhecidas, valores após `senha=`/`token=`/`api_key=` e a pasta pessoal (`~`). Os logs usam `RedactingFormatter` (mensagem e traceback), e `httpx`/`httpcore` ficam em `WARNING`.
  - O comando `sentinel` roda por `cli.run`: erro inesperado mostra só a família e o tipo (`Falha ao acessar o Postgres da memória (OperationalError)`), nunca a mensagem original. Erro de configuração lista os **nomes** dos campos inválidos, não os valores. `SENTINEL_DEBUG=1` mostra o traceback sem variáveis locais e redigido. O Typer roda com `pretty_exceptions_enable=False`.
  - Texto vindo do banco (`changes.description`) passa por `clean_text` antes de chegar ao prompt e ao terminal: sem caracteres de controle e de formatação (ANSI, bidi), um espaço entre palavras e no máximo 200 caracteres. O prompt do analista diz que os fatos são dados, não instruções.
  - O erro do `OllamaEmbedder` não repete a mensagem do `httpx` (ela pode trazer URL com credencial).
  - Higiene automática: `tests/test_repo_hygiene.py` falha se um arquivo privado, um segredo ou um caminho pessoal estiver versionado ou for entrar num `git add .`. O relatório diz arquivo e linha, nunca o valor.
- **Regra de trabalho:** antes de qualquer preparação de commit, rodar `pytest`, conferir `git status` e `git diff` e checar log, traceback e mensagens de erro do que mudou. Todo log novo (por exemplo o `audit.jsonl` da etapa 5 e o Telegram da etapa 7) usa `redact` e nunca grava DSN, token nem a URL do bot.
- **Auditado e sem achado:** todo SQL do código usa parâmetros ou colunas constantes; a conexão com os eventos é `mode=ro` com `query_only`; `pip-audit` não encontrou vulnerabilidade conhecida nas dependências.
- **Limite conhecido:** a redação por padrão não pega um segredo sem formato conhecido que não esteja registrado. Por isso o `Settings` registra tudo o que é secreto, e novos segredos devem entrar em `secret_values()`.
- **Descartado:** confiar no `repr` do Pydantic sem `SecretStr`; mostrar a mensagem original do erro "só em desenvolvimento" (acaba indo para issues públicas); remover o traceback por completo (atrapalha quem testa).

## 19. Publicação

- `README.md`, `LICENSE` (MIT), `SECURITY.md`, CI (`.github/workflows/ci.yml`, permissão só de leitura, sem senha: o Postgres do job aceita conexão local sem senha) e Dependabot (pip e GitHub Actions).
- O `.gitignore` ignora só `/data/` e `/logs/` da raiz: a regra antiga `data/` escondia o pacote `sentinel/data/` (gerador e conexão), que por isso nunca tinha sido versionado.
- O `CLAUDE.md` e o `data-sentinel-CLAUDE.md` seguem fora do Git; o README e este arquivo são a documentação pública.
