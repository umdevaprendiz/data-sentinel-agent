# Política de segurança

## Como reportar uma vulnerabilidade

**Não abra uma issue pública** para uma falha de segurança. Use o relato privado do GitHub: na página do repositório, aba **Security** → **Report a vulnerability**. Descreva o que acontece, como reproduzir e o impacto. Não inclua senhas, tokens ou dados reais no relato.

## O que o projeto protege

- **Segredos** (`MEMORY_DSN`, `TELEGRAM_BOT_TOKEN`) vivem só no `.env`, que fica fora do Git, e são tratados como `SecretStr`.
- **Logs, tracebacks e mensagens de erro** não devem conter segredos, variáveis locais nem o caminho da pasta pessoal. Se você encontrar um caso em que isso acontece, é uma vulnerabilidade: reporte.
- **Os dados nunca são alterados**: a conexão com o SQLite é somente leitura.
- **Nenhuma ação real**: as ações de autonomia L3 rodam só num simulador.
- **Telegram**: só o chat configurado em `TELEGRAM_CHAT_ID` pode aprovar ou ignorar uma proposta.

## Fora de escopo

- Qualidade ou segurança das respostas de um modelo de linguagem escolhido por você (a camada de opinião é declaradamente não verificada).
- Vulnerabilidades em dependências de terceiros já conhecidas e sem exploração específica neste projeto (o Dependabot acompanha essas).

## Se você publicar um fork

Nunca versione o `.env`, bancos (`*.db`), a pasta `logs/` nem arquivos de chave. O teste `tests/test_repo_hygiene.py` e o `.gitignore` existem para impedir isso; mantenha-os.
