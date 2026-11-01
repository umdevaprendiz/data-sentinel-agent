# DataSentinel

Agente autônomo que **vigia métricas de negócio, detecta anomalias, investiga a causa sozinho, dá uma opinião própria e age conforme o nível de autonomia que conquistou**. Tudo o que ele afirma como *fato* é conferido por código.

É um projeto de engenharia de IA. Os dados são **sintéticos**, gerados pelo próprio projeto com a causa de cada anomalia **injetada de propósito**, o que dá um gabarito para medir se o agente acertou.

> **O LLM propõe e opina; o código executa, mede e veta.**
> O modelo escolhe consultas, levanta hipóteses e escreve opiniões. Todo número vem de uma consulta executada pelo código, e uma hipótese só vira causa se o código a confirmar. O modelo pode errar na opinião, mas não consegue inventar um fato.

## Regras do produto

| Regra | Como o código garante |
|---|---|
| Nunca altera os dados | Conexão somente leitura (`mode=ro` no SQLite) |
| Todo número do relatório veio de uma consulta | O texto do LLM usa *placeholders* (`{f003.observed}`) que o código preenche; dígito solto é recusado |
| Causa só existe se o código confirmar | `test_hypothesis` mede o poder de explicação; abaixo do limiar a hipótese não vira causa |
| Fato e opinião nunca se misturam | O relatório tem duas camadas: **fatos** (verificados) e **opinião do agente** (rotulada, com confiança) |
| Prefere "não identificada" a inventar | `inconclusive` é um resultado válido |
| Nenhuma ação real | Ações de autonomia L3 rodam só num **simulador** |
| Nada sai da máquina | LLM local (Ollama); só o Telegram usa rede, e só para alerta e aprovação |

Níveis de autonomia: **L1** relatório · **L2** propõe uma ação e espera aprovação humana · **L3** executa sozinho só ações reversíveis e de baixo risco, no simulador.

## Estado atual

| Etapa | Entrega | Estado |
|---|---|---|
| 1 | Contratos, configuração, gerador de dados com causa injetada, 20 cenários | ✅ |
| 2 | Detector de anomalias com início estimado e métrica-raiz | ✅ |
| 3 | Menu de ferramentas, drill-down com efeito de mistura, plano determinístico | ✅ |
| 4 | Verificador, placeholders, narrador de template e analista (testado com um LLM falso) | ✅ |
| — | Memória de incidentes e busca de contexto com pgvector | ⬜ em construção |
| 5 | Grafo (LangGraph), memória e log de auditoria | ⬜ |
| 6 | Investigador conduzido pelo LLM (Ollama) | ⬜ |
| 7 | Aprovação pelo Telegram (L2) | ⬜ |
| 8 | Política de autonomia e simulador de operações (L3) | ⬜ |
| 9 | Benchmark completo | ⬜ |

**Ainda não há números de desempenho publicados.** Eles só entram aqui depois de rodar o benchmark (etapa 9). As decisões de projeto e os limiares estão em [docs/DECISIONS.md](docs/DECISIONS.md).

## Como rodar

Requisitos: Python 3.12 ou superior. O Ollama só é necessário nas etapas com LLM (ainda não ligadas ao fluxo principal).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; no Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # edite só se precisar; o .env nunca vai para o Git
```

```bash
sentinel generate --scenario s01   # gera os dados com a causa injetada
sentinel detect                    # só o detector
sentinel investigate               # detector + investigação + relatório de template
```

```bash
pytest                  # testes rápidos, sem modelo e sem banco externo
pytest -m slow          # calibração em muitas sementes (alguns minutos)
ruff check . && ruff format --check .
```

## Configuração e segredos

Toda a configuração fica no `.env` (veja o [.env.example](.env.example), que **não tem nenhum valor secreto**).

- **Cada pessoa usa o próprio ambiente.** Para o Telegram, crie o seu bot e ponha o token em `TELEGRAM_BOT_TOKEN` e o seu chat em `TELEGRAM_CHAT_ID`; o agente só aceita aprovações desse chat. Para a memória, aponte `MEMORY_DSN` para um Postgres seu (com a extensão `pgvector`).
- Token e senha do banco são lidos como `SecretStr`: não aparecem em `repr`, log nem traceback.
- Se `MEMORY_DSN` ficar vazio, a memória funciona só em RAM e nada é gravado em disco.

## Segurança e privacidade

- **Erros sem dados.** O comando `sentinel` mostra só o tipo do erro, nunca a mensagem original nem variáveis locais. Logs e tracebacks passam por redação de senhas, tokens e do caminho da sua pasta pessoal.
- **Modo de depuração.** `SENTINEL_DEBUG=1` mostra o traceback (sem variáveis locais e com segredos redigidos). Leia-o antes de colar em uma issue pública.
- **Texto do banco é dado, não instrução.** Descrições de `changes` entram no prompt do LLM e no terminal depois de limpas (sem caracteres de controle, com tamanho limitado), e o verificador recusa número, id e fatia que o código não confirmou. Mesmo assim, a camada de *opinião* é não verificada: trate-a como sugestão.
- **Higiene do repositório.** Um teste (`tests/test_repo_hygiene.py`) falha se um arquivo privado, um segredo ou um caminho pessoal estiver prestes a ser versionado.
- Para reportar uma vulnerabilidade, veja [SECURITY.md](SECURITY.md).

## Limites conhecidos

- Dados **sintéticos**; os cenários e as magnitudes são pontos de partida, não dados reais.
- Granularidade diária e uma métrica-raiz por incidente.
- Modelo local pequeno e CPU lenta: incidentes no modo com LLM levam minutos.
- Uma deriva lenta e fraca (cenário s20) nem sempre é detectada; o benchmark vai relatar isso como está.
- Ações só no simulador. O projeto não altera nenhum sistema real.
- O verificador não vê número por extenso ("três causas") nem negação ("não caiu").

## Licença

[MIT](LICENSE).
