# README - Processador de Certificados Acadêmicos com OCR e NER (Google Colab)

## 📌 Visão Geral

Script Python para rodar no Google Colab que processa certificados acadêmicos (PDF/imagems) e gera relatórios de horas complementares automaticamente.

## 🔧 Funcionalidades

- Extrai texto de PDFs e imagens (OCR)
- Identifica automaticamente (NER e REGEX):
  - Nome do aluno
  - Instituição
  - Atividade
  - Carga horária
  - Data
- Classifica em categorias acadêmicas
- Gera relatório consolidado

## 🚀 Como Usar no Colab

1. Acesse [Google Colab](https://colab.research.google.com/)
2. Faça upload deste notebook
3. Monte seu Google Drive (`/content/drive/MyDrive/Certificados`)
4. Cole seus certificados na pasta `Certificados`
5. Execute todas as células (Runtime > Run all)

## 📂 Saída

- `resultados_processamento.json`: dados extraídos
- `relatorios.txt`: consolidação por aluno

## ⚠️ Requisitos

- Certificados com texto legível
- Arquivos em PDF, JPG ou PNG
