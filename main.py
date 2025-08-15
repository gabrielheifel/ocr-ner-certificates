# Instale as dependências necessárias (execute esta célula primeiro)

# Atualiza os repositórios e instala Tesseract + Poppler (para PDF)
!apt-get update
!apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils

# Instala as bibliotecas Python necessárias
!pip install pytesseract Pillow pdf2image langdetect

# Instala os modelos grandes do spaCy para português e inglês
!python -m spacy download pt_core_news_lg
!python -m spacy download en_core_web_lg

# =============================================================================
# IMPORTS E CONFIGURAÇÕES
# =============================================================================

import subprocess
import os
import spacy
import json
from typing import Dict, List
import pandas as pd
import re
from langdetect import detect
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from google.colab import drive

# Monta o Google Drive
drive.mount('/content/drive', force_remount=True)
CERTIFICADOS_PATH = '/content/drive/MyDrive/Certificados'

# Mapeamento único de categorias
palavras_chave_atividades = {
    "Monitorias": [
        "monitoria", "monitor", "monitora", "atividade de monitoria",
        "monitoring", "teaching assistant", "mentor", "assistant activity"
    ],
    "Bolsista/Voluntário de Projetos de Pesquisa": [
        "bolsista de pesquisa", "projeto de pesquisa", "pesquisa", "voluntário de pesquisa", "iniciação científica",
        "bolsista de projeto de pesquisa", "voluntário de projeto de pesquisa",
        "research grant", "research project", "scientific research", "volunteer research", "undergraduate research"
    ],
    "Bolsista/Voluntário de Projetos de Extensao": [
        "bolsista de extensão", "projeto de extensão", "extensão universitária", "voluntário de extensão",
        "bolsista de projeto de extensão", "voluntário de projeto de extensão",
        "extension project", "university extension", "extension volunteer", "extension program"
    ],
    "Bolsista/Voluntario de Projetos de Ensino": [
        "bolsista de ensino", "projeto de ensino", "ensino", "voluntário de ensino",
        "bolsista de projeto de ensino", "voluntário de projeto de ensino",
        "teaching project", "teaching volunteer", "teaching program", "educational project"
    ],
    "Participação em Atividades de Extensão (como organizador, colaborador ou ministrante)": [
        "atividade de extensão", "organizador", "colaborador", "ministrante", "extensão", "participação em extensão",
        "extension activity", "organizer", "collaborator", "speaker", "extension participation"
    ],
    "Participação em Semana Acadêmica do Curso de Computação": [
        "sacomp", "semana acadêmica", "semana do curso", "semana de computação", "semana acadêmica de computação",
        "academic week", "course week", "computer science week", "cs academic week"
    ],
    "Participação em Cursos e Escolas": [
        "curso", "escola", "participação em curso", "participação em escola", "curso de", "escola de",
        "course", "school", "participation in course", "participation in school", "course on", "school on"
    ],
    "Participação em Evento Científico": [
        "evento científico", "simpósio", "congresso", "jornada", "encontro científico", "workshop", "seminário", "evento",
        "scientific event", "symposium", "conference", "scientific meeting", "workshop", "seminar", "event"
    ],
    "Publicação de Artigo Científico": [
        "artigo científico", "publicação", "publicado", "artigo", "paper", "revista científica",
        "scientific article", "publication", "published", "article", "paper", "scientific journal"
    ],
    "Representação Estudantil": [
        "representante estudantil", "representação estudantil", "diretório acadêmico", "centro acadêmico", "representante de turma",
        "student representative", "academic representation", "student council", "class representative"
    ],
    "Obtenção de Prêmios e Distinções": [
        "prêmio", "distinção", "menção honrosa", "premiado", "destaque", "honraria",
        "award", "distinction", "honorable mention", "awarded", "highlight", "honor"
    ],
    "Certificações Profissionais": [
        "certificação profissional", "certificado profissional", "certificação", "profissionalizante",
        "professional certification", "certified professional", "certification", "professional training"
    ]
}

INSTITUTION_KEYWORDS_PT = ['universidade', 'escola', 'instituto', 'faculdade', 'colégio', 'ifsul', 'ifrs', 'ifsp', 'federal', 'senac', 'senai', 'ufpel']
INSTITUTION_KEYWORDS_EN = ['university', 'school', 'institute', 'college', 'ifsp', 'senac', 'senai', 'federal', 'rocketseat']
ACTIONS_KEYWORDS_PT = ['curso', 'curso de', 'palestra', 'treinamento', 'oficina', 'workshop', 'formação', 'seminário', 'minicurso', 'ouvinte', 'bolsista', 'artigo', 'apresentador', 'participação', 'comissão organizadora', 'organizador', 'participou', 'prêmio', 'melhor artigo', 'melhores artigos']
ACTIONS_KEYWORDS_EN = ['course', 'lecture', 'training', 'workshop', 'formation', 'seminar', 'mini-course', 'attendee', 'scholarship', 'article', 'presenter', 'participation', 'organizing committee', 'organizer', 'participated']

print("✅ Configurações carregadas!")

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def load_json(path):
    """Carrega arquivo JSON"""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_json(data, path):
    """Salva arquivo JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clean_text(text: str) -> str:
    """Limpa o texto extraído"""
    # Remove espaços extras em cada linha
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
    # Remove linhas totalmente vazias
    lines = [line for line in lines if line]
    # Junta as linhas com \n preservando quebras de linha
    text = '\n'.join(lines)
    # Remove caracteres indesejados, mantendo letras, números, pontuação e \n
    text = re.sub(r'[^\w\sÀ-ÿ.,;:!?/\-\n]', '', text)
    # Corrige palavras em maiúsculas para Title Case
    def fix_caps(match):
        word = match.group(0)
        return word.title()
    text = re.sub(r'\b[A-ZÀ-Ý]{3,}\b', fix_caps, text)
    return text


print("✅ Funções utilitárias definidas!")

# =============================================================================
# FUNÇÕES DE EXTRAÇÃO DE ENTIDADES
# =============================================================================

def extract_institution_combined(tokens, text, keywords, lang='pt'):
    """Extrai a instituição combinando regex e entidades spaCy, escolhendo a melhor candidata"""

    if lang == 'pt':
        patterns = [
            r'Universidade\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'Universidade\s+Federal\s+de\s+[A-Z][a-z]+',
            r'Instituto\s+Federal\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'Faculdade\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'Escola\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'Col[eé]gio\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'IFSUL', r'IFRS', r'IFSP', r'SENAC', r'SENAI'
        ]
    else:  # English
        patterns = [
            r'Federal\s+University\s+of\s+[A-Z][a-z]+',
            r'University\s+of\s+[A-Z][a-z]+',
            r'Institute\s+of\s+[A-Z][a-z]+',
            r'College\s+of\s+[A-Z][a-z]+',
            r'School\s+of\s+[A-Z][a-z]+',
            r'IFSP', r'SENAC', r'SENAI', r'Rocketseat'
        ]

    candidates = []
    found_by_regex = False
    found_by_ner = False

    # Regex - todas as matches
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            institution_found = match.group(0).strip()
            if institution_found not in candidates:
                candidates.append(institution_found)
                found_by_regex = True

    # Entidades spaCy com keywords
    for ent, label in tokens['entidades']:
        if label in ['ORG', 'LOC', 'MISC']:
            ent_lower = ent.lower()
            if any(kw in ent_lower for kw in keywords):
                if ent not in candidates:
                    candidates.append(ent)
                    found_by_ner = True

    # Função simples de score para escolher melhor candidato
    def score(name):
        name_lower = name.lower()
        count_kw = sum(kw in name_lower for kw in keywords)
        return count_kw * 10 + len(name)  # peso em keywords + comprimento

    if candidates:
        candidates.sort(key=score, reverse=True)
        best = candidates[0]
    else:
        best = None

    # Normalização para Udemy
    if re.search(r'ude\.?my', text, re.IGNORECASE):
        best = "Udemy"

    if best:
        if found_by_regex and not found_by_ner:
            print(f"INFO: Instituição '{best}' capturada via REGEX")
        elif found_by_ner and not found_by_regex:
            print(f"INFO: Instituição '{best}' capturada via NER")
        elif found_by_regex and found_by_ner:
            print(f"INFO: Instituição '{best}' capturada via REGEX e NER")
        else:
            print(f"INFO: Instituição '{best}' capturada, origem desconhecida")

    return best

def extract_action_combined(tokens, text, keywords, lang='pt'):
    candidates = []
    found_by_regex = False
    found_by_ner = False

    # Padrões simples de regex para capturar expressões comuns de ação
    patterns = []
    if lang == 'pt':
        patterns = [rf'{kw}[^.,;]*' for kw in keywords if len(kw) > 2]  # evita keywords muito curtas
    else:
        patterns = [rf'{kw}[^.,;]*' for kw in keywords if len(kw) > 2]

    # Busca candidatos por regex
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(0).strip()
            if candidate not in candidates:
                candidates.append(candidate)
                found_by_regex = True

    # Busca em entidades NER com labels plausíveis para ações
    ner_labels = ['MISC', 'WORK_OF_ART', 'EVENT', 'ORG']
    for ent, label in tokens['entidades']:
        if label in ner_labels:
            ent_lower = ent.lower()
            if any(kw in ent_lower for kw in keywords):
                if ent not in candidates:
                    candidates.append(ent)
                    found_by_ner = True

    # Score simples para escolher melhor candidato: peso para mais keywords + tamanho
    def score(name):
        name_lower = name.lower()
        count_kw = sum(kw in name_lower for kw in keywords)
        return count_kw * 10 + len(name)

    if candidates:
        candidates.sort(key=score, reverse=True)
        best = candidates[0]

        # Print de origem da detecção
        if found_by_regex and not found_by_ner:
            print(f"INFO: Ação '{best}' capturada via REGEX")
        elif found_by_ner and not found_by_regex:
            print(f"INFO: Ação '{best}' capturada via NER")
        elif found_by_regex and found_by_ner:
            print(f"INFO: Ação '{best}' capturada via REGEX e NER")
        else:
            print(f"INFO: Ação '{best}' capturada, origem desconhecida")

        return best

    return None

def is_valid_name(name: str) -> bool:
    words = name.strip().split()
    if len(words) < 2:
        return False
    if any(not w[0].isupper() for w in words):
        return False
    if any(any(c.isdigit() for c in w) for w in words):
        return False
    invalid_words = {'Prof', 'Dr', 'Sr', 'Sra', 'Ms'}
    if any(w in invalid_words for w in words):
        return False
    return True

def extract_name_with_context(tokens):
    """Extrai nome usando contexto e filtrando por palavras-chave válidas e inválidas."""
    context_keywords = ['certificamos que', 'aluno', 'participou', 'concluiu']
    invalid_keywords = ['coordenador', 'professor', 'coordinator', 'director', 'instructor']

    name_candidates = []

    for sentence in tokens['frases']:
        sentence_lower = sentence.lower()

        # Pular frases com cargos inválidos
        if any(invalid in sentence_lower for invalid in invalid_keywords):
            continue

        # Verificar se a frase contém alguma palavra-chave de contexto
        if any(keyword in sentence_lower for keyword in context_keywords):
            for ent, label in tokens['entidades']:
                if label == 'PER' and ent in sentence:
                    name_candidates.append(ent)

    if name_candidates:
        return name_candidates[0]
    return None

def extract_name_with_context_en(tokens):
    """Extracts person name using context keywords in English"""
    context_keywords = ['certify that', 'student', 'attended', 'completed', 'participated']
    invalid_keywords = ['coordinator','professor','director','instructor','manager','supervisor','teacher','head']

    name_candidates = []

    for sentence in tokens['frases']:
        sentence_lower = sentence.lower()

        # Pular frases com cargos inválidos
        if any(invalid in sentence_lower for invalid in invalid_keywords):
            continue

        # Verificar se a frase contém alguma palavra-chave de contexto
        if any(keyword in sentence_lower for keyword in context_keywords):
            for ent, label in tokens['entidades']:
                if label == 'PERSON' and ent in sentence:
                    name_candidates.append(ent)

    if name_candidates:
        return name_candidates[0]
    return None

def extract_user_name(tokens, lang):
    if lang == 'pt':
        name_context = extract_name_with_context(tokens)
        if name_context and is_valid_name(name_context):
            return name_context
        else:
            for ent, label in tokens['entidades']:
                if label == 'PER' and is_valid_name(ent):
                    return ent
    else:  # inglês
        name_context = extract_name_with_context_en(tokens)
        if name_context and is_valid_name(name_context):
            return name_context
        else:
            for ent, label in tokens['entidades']:
                if label == 'PERSON' and is_valid_name(ent):
                    return ent
    return None

print("✅ Funções de extração definidas!")

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO PRINCIPAL
# =============================================================================

def load_spacy_models():
    """Carrega modelos spaCy"""
    try:
        nlp_pt = spacy.load('pt_core_news_lg')
    except OSError:
        print("Baixando modelo de linguagem em português...")
        print("Execute: !python -m spacy download pt_core_news_lg")
        print("\n⚠️ IMPORTANTE: Por favor, reinicie o kernel/runtime e execute o código novamente!")
        raise Exception("Reinicie o kernel para carregar o modelo spaCy")
    try:
        nlp_en = spacy.load('en_core_web_lg')
    except OSError:
        print("Baixando modelo de linguagem em inglês...")
        print("Execute: !python -m spacy download en_core_web_lg")
        print("\n⚠️ IMPORTANTE: Por favor, reinicie o kernel/runtime e execute o código novamente!")
        raise Exception("Reinicie o kernel para carregar o modelo spaCy em inglês")
    print("✅ Modelos spaCy carregados!")
    return nlp_pt, nlp_en

def process_text(nlp, text: str) -> Dict:
    """Processa texto com spaCy"""
    doc = nlp(text)
    tokens = {
        'entidades': [(ent.text, ent.label_) for ent in doc.ents],
        'frases': [sent.text for sent in doc.sents],
        'palavras_chave': [token.text for token in doc if not token.is_stop and not token.is_punct],
        'original_text': text
    }
    print(tokens)
    # Print the size of the 'entidades' array
    print(f"Tamanho do array 'entidades': {len(tokens['entidades'])}")
    return tokens

def extract_info_pt(tokens):
    """Extrai informações de texto em português"""
    result = {
        'user_name': None,
        'action': None,
        'institution': None,
        'dates': [],
        'hours': [],
        'detected_categories': []
    }

    # Name extraction with context
    result['user_name'] = extract_user_name(tokens, lang='pt')

    # Institution extraction with regex and context
    result['institution'] = extract_institution_combined(tokens, tokens['original_text'], INSTITUTION_KEYWORDS_PT, lang='pt')

    # Action extraction
    result['action'] = extract_action_combined(tokens, tokens['original_text'], ACTIONS_KEYWORDS_PT, lang='pt')

    # Date and Hour extraction
    text = tokens['original_text']
    for ent, label in tokens['entidades']:
        if label == 'DATE' and ent not in result['dates']:
            result['dates'].append(ent)

    # Regex para datas específicas e formatos
    date_pattern = (
        r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b'
        r'|'
        r'de \d{1,2} a \d{1,2} de [a-zçãé]+ de \d{4}'
        r'|'
        r'entre \d{1,2} e \d{1,2} de [a-zçãé]+ de \d{4}'
        r'|'
        r'\d{1,2} a \d{1,2} de [a-zçãé]+ de \d{4}'
        r'|'
        r'\d{1,2} de [a-zçãé]+ de \d{4}'
    )
    dates_regex = re.findall(date_pattern, text, re.IGNORECASE)
    for date_str in dates_regex:
        if date_str not in result['dates']:
            # Processa intervalos para separar em duas datas
            match = re.match(r'(?:de )?(\d{1,2}) a (\d{1,2}) de ([a-zçãé]+) de (\d{4})', date_str, re.IGNORECASE)
            if match:
                day_start, day_end, month, year = match.groups()
                d1 = f"{int(day_start):02d} de {month} de {year}"
                d2 = f"{int(day_end):02d} de {month} de {year}"
                if d1 not in result['dates']:
                    result['dates'].append(d1)
                if d2 not in result['dates']:
                    result['dates'].append(d2)
            else:
                result['dates'].append(date_str)

    for ent, label in tokens['entidades']:
        if label == 'TIME' and ent not in result['hours']:
            result['hours'].append(ent)

    # Regex para horas
    hours_regex = re.findall(r'\b\d{1,5}\s?(?:h|horas?|hora\(s\)?)\b', text, re.IGNORECASE)
    hours_hhmm = re.findall(r'\b\d{1,5}:00\b', text)
    for h in hours_regex + hours_hhmm:
        if h not in result['hours']:
            result['hours'].append(h)

    # Category detection
    text_lower = text.lower()
    detected_categories = set()
    if 'ouvinte' in text_lower:
        detected_categories.add('ouvinte')
    else:
        for category, keywords in palavras_chave_atividades.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected_categories.add(category)
                    break
    result['detected_categories'] = list(detected_categories)

    return result

def extract_info_en(tokens):
    """Extrai informações de texto em inglês"""
    result = {
        'user_name': None,
        'action': None,
        'institution': None,
        'dates': [],
        'hours': [],
        'detected_categories': []
    }

    # Name extraction
    result['user_name'] = extract_user_name(tokens, lang='en')

    # Institution extraction
    result['institution'] = extract_institution_combined(tokens, tokens['original_text'], INSTITUTION_KEYWORDS_EN, lang='en')

    # Action extraction
    result['action'] = extract_action_combined(tokens, tokens['original_text'], ACTIONS_KEYWORDS_EN, lang='en')

    # Dates and hours extraction
    text = tokens['original_text']

    # Datas de NER
    for ent, label in tokens['entidades']:
        if label == 'DATE' and ent not in result['dates']:
            result['dates'].append(ent)

    # Datas regex
    date_pattern = (
        r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b'
        r'|'
        r'from \d{1,2} to \d{1,2} [a-z]+ \d{4}'
        r'|'
        r'between \d{1,2} and \d{1,2} [a-z]+ \d{4}'
        r'|'
        r'\d{1,2} to \d{1,2} [a-z]+ \d{4}'
        r'|'
        r'\d{1,2} [a-z]+ \d{4}'
    )
    dates_regex = re.findall(date_pattern, text, re.IGNORECASE)
    for d in dates_regex:
        if d not in result['dates']:
            result['dates'].append(d)

    # Horas NER
    for ent, label in tokens['entidades']:
        if label == 'TIME' and ent not in result['hours']:
            result['hours'].append(ent)

    # Horas regex
    hours_regex = re.findall(r'\b\d{1,5}\s?(?:h|hours?)\b', text, re.IGNORECASE)
    hours_hhmm = re.findall(r'\b\d{1,5}:00\b', text)
    for h in hours_regex + hours_hhmm:
        if h not in result['hours']:
            result['hours'].append(h)

    # Category detection
    text_lower = text.lower()
    detected_categories = set()
    if 'listener' in text_lower:
        detected_categories.add('ouvinte')
    else:
        for category, keywords in palavras_chave_atividades.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected_categories.add(category)
                    break
    result['detected_categories'] = list(detected_categories)

    return result

print("✅ Funções de processamento definidas!")

# =============================================================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =============================================================================

def process_json_file(nlp_pt, nlp_en, file_path: str) -> Dict:
    """Processa um arquivo JSON contendo um array de objetos com campo 'text'"""
    resultados = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            conteudo = f.read()
            data = json.loads(conteudo)

        if not isinstance(data, dict) or 'processedFiles' not in data:
            print(f"Erro: O arquivo não contém o campo 'processedFiles'. Campos disponíveis: {list(data.keys())}")
            return {}

        processed_files = data['processedFiles']
        if not isinstance(processed_files, list):
            print(f"Erro: O campo 'processedFiles' não é um array. Tipo encontrado: {type(processed_files)}")
            return {}

        encontrou_texto = False

        for i, item in enumerate(processed_files):
            if 'text' in item and item['text']:
                encontrou_texto = True
                print(f"\nProcessing text from file: {item['fileName']}")
                print("\nOriginal text:")
                print("-" * 50)
                print(item['text'])
                print("-" * 50)

                # Clean the text before processing
                cleaned_text = clean_text(item['text'])

                # Detect language using langdetect
                try:
                    language = detect(cleaned_text)
                except Exception:
                    language = 'pt'  # fallback

                if language == 'en':
                    #print("Detected language: English (en). Using spaCy English model.")
                    tokens = process_text(nlp_en, cleaned_text)
                    info_extracted = extract_info_en(tokens)
                else:
                    #print(f"Detected language: {language}. Using spaCy Portuguese model.")
                    tokens = process_text(nlp_pt, cleaned_text)
                    info_extracted = extract_info_pt(tokens)

                # Adiciona o texto limpo ao dicionário de resultados
                info_extracted['cleaned_text'] = cleaned_text

                # Adiciona o link ao arquivo processado
                info_extracted['file_link'] = f"drive.com/{item['fileName']}"

                print("\nExtracted information from this file:")
                print("-" * 50)
                for key, value in info_extracted.items():
                    print(f"{key}: {value}")

                resultados[item['fileName']] = info_extracted

        if not encontrou_texto:
            print("Nenhum texto válido encontrado para processar")

        return resultados

    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        print("Verifique se o arquivo está em formato JSON válido")
        return {}
    except Exception as e:
        print(f"Erro ao processar arquivo {file_path}: {e}")
        return {}

def save_results(results: Dict, output_path: str):
    """Salva os resultados em um arquivo JSON"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Resultados salvos em: {output_path}")
    except Exception as e:
        print(f"Erro ao salvar resultados: {e}")
        print("Verifique se você tem permissão para escrever no diretório.")

def extract_text_with_pytesseract(folder_path):
    """Extrai texto de imagens/PDFs usando pytesseract"""
    processed_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        ext = file_name.lower().split('.')[-1] if '.' in file_name else ''
        try:
            if ext in ('png', 'jpg', 'jpeg', 'tiff'):
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img, lang='por')
                processed_files.append({
                    "fileName": file_name,
                    "type": ext,
                    "text": text
                })
            elif ext == 'pdf':
                pages = convert_from_path(file_path)
                text = ""
                for page in pages:
                    text += pytesseract.image_to_string(page, lang='por') + "\n"
                processed_files.append({
                    "fileName": file_name,
                    "type": ext,
                    "text": text
                })
        except Exception as e:
            print(f"Erro ao processar {file_name}: {e}")
    return {
        "totalFiles": len(processed_files),
        "processedFiles": processed_files
    }

print("✅ Função principal de processamento definida!")

# =============================================================================
# FUNÇÃO CRIAR RELATÓRIOS
# =============================================================================

# Mapeamento de categorias resumidas
CATEGORIAS_PRINCIPAIS = {
    'Ensino': [
        "Monitorias",
        "Bolsista/Voluntario de Projetos de Ensino"
    ],
    'Pesquisa': [
        "Bolsista/Voluntário de Projetos de Pesquisa",
        "Publicação de Artigo Científico"
    ],
    'Extensão': [
        "Bolsista/Voluntário de Projetos de Extensao",
        "Participação em Atividades de Extensão (como organizador, colaborador ou ministrante)",
        "Participação em Semana Acadêmica do Curso de Computação",
        "Participação em Evento Científico",
        "Representação Estudantil",
        "Obtenção de Prêmios e Distinções",
        "Certificações Profissionais"
    ],
    'Livres': [
        "Participação em Cursos e Escolas",
    ]
}

def normalize_name(name: str) -> str:
    """Simplifica nome para comparação: minúsculo e sem espaços extras"""
    return ' '.join(name.strip().lower().split())

def parse_hours(hours_list):
    """Converte lista de strings de horas para float total (horas)"""
    total = 0.0
    for h in hours_list:
        h = h.lower().strip()
        if ':' in h:
            parts = h.split(':')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                total += int(parts[0]) + int(parts[1])/60
        else:
            match = re.search(r'(\d+(?:[\.,]\d+)?)', h)
            if match:
                val = match.group(1).replace(',', '.')
                try:
                    total += float(val)
                except:
                    pass
    return total

def categorize_hours(cert):
    """Recebe um certificado e retorna um dicionário com horas por categoria principal"""
    horas_total = parse_hours(cert.get('hours', []))
    categorias_detected = cert.get('detected_categories', [])

    horas_por_categoria = {
        'Ensino': 0.0,
        'Pesquisa': 0.0,
        'Extensão': 0.0,
        'Livres': 0.0
    }

    for cat in categorias_detected:
        # Padronizações especiais por categoria
        if cat == "Participação em Evento Científico":
            horas_adicionais = 17.0
            horas_por_categoria['Extensão'] += horas_adicionais
        elif cat == "Publicação de Artigo Científico":
            horas_adicionais = 34.0
            horas_por_categoria['Pesquisa'] += horas_adicionais
        elif cat == "Obtenção de Prêmios e Distinções":
            horas_adicionais = 68.0
            horas_por_categoria['Extensão'] += horas_adicionais
        else:
            # Para categorias normais, adiciona o total extraído
            for main_cat, subcats in CATEGORIAS_PRINCIPAIS.items():
                if cat in subcats:
                    horas_por_categoria[main_cat] += horas_total
                    break

    # Se não tiver categoria, considera como Formação Livre
    if sum(horas_por_categoria.values()) == 0:
        horas_por_categoria['Livres'] += horas_total

    return horas_por_categoria

def aggregate_certificates(certificates):
    aggregation = {}
    for cert in certificates:
        name = cert.get('user_name')
        if not name:
            continue
        norm_name = normalize_name(name)

        horas_cat = categorize_hours(cert)

        if norm_name not in aggregation:
            aggregation[norm_name] = {
                'name': name,
                'total_hours': 0.0,
                'category_hours': {'Ensino':0.0,'Pesquisa':0.0,'Extensão':0.0,'Livres':0.0},
                'certificates': []
            }

        aggregation[norm_name]['total_hours'] += sum(horas_cat.values())
        for key in horas_cat:
            aggregation[norm_name]['category_hours'][key] += horas_cat[key]

        aggregation[norm_name]['certificates'].append(cert)

    return aggregation


def save_report_txt(aggregation, output_path):
    """Salva relatório de certificados em formato de texto (ordem alfabética)"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # Ordena pelo nome normalizado
            for norm_name, data in sorted(aggregation.items(), key=lambda x: x[0]):
                ensino = data['category_hours'].get('Ensino', 0.0)
                pesquisa = data['category_hours'].get('Pesquisa', 0.0)
                extensao = data['category_hours'].get('Extensão', 0.0)
                livres_lancada = data['category_hours'].get('Livres', 0.0)

                limite_atividades_complementar = 320
                limite_formacao_livre = 217

                total_complementares = ensino + pesquisa + extensao
                aproveitadas_complementares = min(total_complementares, limite_atividades_complementar)
                excedente_para_livre = max(total_complementares - limite_atividades_complementar, 0)
                total_formacao_livre = livres_lancada + excedente_para_livre

                # Análise
                analise = []
                if aproveitadas_complementares >= limite_atividades_complementar:
                    analise.append("A carga horária necessária em Atividades Complementares foi obtida.")
                else:
                    analise.append("Ainda faltam horas em Atividades Complementares.")

                if total_formacao_livre >= limite_formacao_livre:
                    analise.append("A carga horária necessária em Formação Livre foi obtida.")
                else:
                    analise.append("Ainda faltam horas em Formação Livre.")

                # Pelo menos 2 categorias nas complementares
                categorias_com_horas = sum(1 for v in [ensino, pesquisa, extensao] if v > 0)
                if categorias_com_horas < 2:
                    analise.append("O aluno precisa ter horas em pelo menos duas categorias de Atividades Complementares.")

                # Relatório
                f.write(f"Extrato de Atividades Complementares\n")
                f.write(f"Nome: {data['name']}\n\n")
                f.write(f"Total de horas: {data['total_hours']:.2f}h\n")
                f.write(f"Ensino: {ensino:.2f}h\n")
                f.write(f"Pesquisa: {pesquisa:.2f}h\n")
                f.write(f"Extensão: {extensao:.2f}h\n\n")
                f.write(f"Total em Atividades Complementares: {total_complementares:.2f}h\n")
                f.write(f"Aproveitadas em Atividades Complementares: {aproveitadas_complementares:.2f}h\n\n")
                f.write(f"Formação Livre (aproveitada da complementar): {excedente_para_livre:.2f}h\n")
                f.write(f"Formação Livre (lançada): {livres_lancada:.2f}h\n")
                f.write(f"Total em Formação Livre: {total_formacao_livre:.2f}h\n\n")
                f.write("**Análise:**\n" + " ".join(analise) + "\n\n")
                f.write("=============================================================================\n\n")
        print(f"Relatório de texto salvo em: {output_path}")
    except Exception as e:
        print(f"Erro ao salvar relatório: {e}")

print("✅ Função relatórios definida!")

# =============================================================================
# FUNÇÃO MAIN E EXECUÇÃO
# =============================================================================

import time

def main():
    """Função principal"""
    nlp_pt, nlp_en = load_spacy_models()
    certificados_path = CERTIFICADOS_PATH
    json_file_path = os.path.join(certificados_path, 'Untitled')

    start_time = time.time()

    try:
        tesseract_version = subprocess.check_output(['tesseract', '--version']).decode('utf-8').splitlines()[0]
    except Exception:
        tesseract_version = "desconhecida"
    print(f"\nExtraindo textos dos certificados com pytesseract na versão {tesseract_version}...")
    data = extract_text_with_pytesseract(certificados_path)
    temp_json_path = os.path.join(certificados_path, 'temp_processed.json')
    with open(temp_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    results = process_json_file(nlp_pt, nlp_en, temp_json_path)
    if results:
        output_path = '/content/drive/MyDrive/resultados_processamento.json'
        save_results(results, output_path)

        agrupados = aggregate_certificates(results.values())
        output_path = '/content/drive/MyDrive/relatorios.txt'
        save_report_txt(agrupados, output_path)
        print("✅ Relatório agregado salvo em relatorios.txt")

    end_time = time.time()
    elapsed_time = end_time - start_time # Calculate the elapsed time
    print(f"\nTempo total para gerar resultados: {elapsed_time:.2f} segundos") # Print the elapsed time


print("✅ Função main definida!")

if __name__ == "__main__":
    main()