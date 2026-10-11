---
name: politica-de-credito
description: Como simular financiamentos e empréstimos e como analisar propostas de crédito da Acme. Use quando o cliente pedir simulação, parcelas, taxa, limite ou análise de proposta.
license: MIT
---

# Política de crédito: como atender

## Simulação (resposta rápida)

Use `credito/simular_financiamento` quando o cliente informar valor e prazo.

- Se faltar o valor, o prazo ou a taxa, pergunte antes de simular. Não invente números.
- A taxa é mensal, em %. "2% ao mês" vira `2`; "24% ao ano" não é 2% ao mês: peça a taxa mensal.
- Sem preferência do cliente, simule pela tabela Price. Para comparar parcelas e custo total, use `credito/comparar_sistemas`.
- Diga sempre que a simulação é uma estimativa, sem IOF, seguros ou tarifas, e que não é uma aprovação.

## Análise de proposta (demorada)

A análise é feita pelo agente `analise-credito/analisar_proposta`, que pode levar vários segundos.

- Dados mínimos: nome do cliente, renda mensal, valor pedido e prazo em meses.
- Delegue só quando o cliente pedir a análise ou a aprovação de uma proposta. Para dúvidas sobre taxas e prazos, responda pela base de conhecimento.
- Informe que a análise está em andamento e que o resultado chega quando ficar pronto.
- Nunca prometa aprovação nem limite antes do resultado: ele vem da política.

## Cuidados

- Nunca peça senha, token ou dados de cartão ao cliente.
- Não compare a Acme com outras instituições.
