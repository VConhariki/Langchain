# Langchain + GitHub Copilot RAG

Este projeto cria um assistente de IA para responder perguntas sobre tributação de fundos de investimento, usando LangChain e o modelo do GitHub Copilot.

A ideia principal é combinar:

- busca de documentos relevantes em um material de referência;
- histórico da conversa;
- instruções do sistema para controlar o comportamento do modelo;
- resposta do modelo com base no contexto disponível.

---

## Estrutura do projeto

- `app.py` — ponto de entrada do programa
- `src/conversa.py` — loop principal da interação com o usuário
- `src/autenticacao.py` — autenticação do GitHub Copilot
- `src/chain.py` — criação da cadeia prompt + modelo
- `src/rag.py` — preparação do banco de conhecimento para busca semântica textual
- `rag.md` — material base usado como contexto
- `config/requirements.txt` — dependências do projeto
- `.env` — arquivo com o token do Copilot (quando gerado)

---

## Como o fluxo funciona

### 1. Início da aplicação

O programa roda em `app.py`:

```python
from conversa import iniciar_conversa


if __name__ == "__main__":
    iniciar_conversa()
```

Esse trecho apenas chama a função responsável por iniciar a conversa.

---

### 2. Autenticação do GitHub Copilot

Arquivo: `src/autenticacao.py`

```python
import os

from dotenv import load_dotenv, set_key
from langchain_githubcopilot_chat import get_copilot_token


def obter_token() -> str:
    load_dotenv()
    copilot_token = os.getenv("COPILOT_TOKEN")

    if not copilot_token or not copilot_token.startswith("tid="):
        print("Iniciando autenticação do GitHub Copilot...")
        copilot_token = get_copilot_token()
        if copilot_token:
            set_key(".env", "COPILOT_TOKEN", copilot_token)

    if not copilot_token:
        raise RuntimeError("Não foi possível obter um token do GitHub Copilot.")

    return copilot_token
```

#### O que acontece aqui?

- `load_dotenv()` lê as variáveis do arquivo `.env`.
- `os.getenv("COPILOT_TOKEN")` tenta recuperar o token já salvo.
- Se não existir ou não for válido, chama `get_copilot_token()` para autenticar com o GitHub Copilot.
- Quando obtém um token, salva no `.env` usando `set_key`.
- Se não conseguir obter o token, o programa gera um erro.

#### Pacotes externos usados

- `python-dotenv`
  - `load_dotenv()` lê variáveis do arquivo `.env`
  - `set_key()` salva uma variável no `.env`

- `langchain_githubcopilot_chat`
  - `get_copilot_token()` executa o processo de autenticação do Copilot

---

### 3. Criação do contexto de busca (RAG)

Arquivo: `src/rag.py`

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


def obter_rag(caminho_arquivo: str | None = None) -> BM25Retriever:
    documentos = TextLoader(caminho_arquivo, encoding="utf-8").load()
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    documentos_divididos = divisor.split_documents(documentos)
    retriever = BM25Retriever.from_documents(documentos_divididos, k=4)
    return retriever
```

#### O que acontece here?

- `TextLoader` lê o arquivo `rag.md` como um documento do LangChain.
- `RecursiveCharacterTextSplitter` divide o texto em trechos menores.
- `BM25Retriever.from_documents(...)` cria um mecanismo de busca que localiza os blocos mais relevantes para a pergunta do usuário.

#### Por que dividir o texto?

Porque enviar todo o conteúdo de uma vez para o modelo pode ser muito grande, lento e pouco eficiente. Dividir em chunks ajuda:

- reduzir custo de tokens;
- melhorar a busca por partes relevantes;
- manter a resposta focada no contexto necessário.

#### Pacotes externos usados

- `langchain_community.document_loaders.TextLoader`
  - carrega arquivos de texto para documentos do LangChain

- `langchain_text_splitters.RecursiveCharacterTextSplitter`
  - quebra textos longos em blocos menores

- `langchain_community.retrievers.BM25Retriever`
  - busca os trechos mais relevantes com base em palavras-chave e relevância

---

### 4. Montagem da cadeia do modelo

Arquivo: `src/chain.py`

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_githubcopilot_chat import ChatGithubCopilot


def criar_chain(copilot_token: str):
    template = """
        Você é um assistente de IA que ajudará com as dúvidas sobre tributação de fundos de investimento.

        Qualquer pergunta fora do tema de tributação de fundos de investimento será ignorada.
        Responda somente com base no contexto fornecido. Se a resposta não estiver no contexto,
        informe que essa informação não consta no material disponível.

        Sempre dê a resposta mais completa possível.
        
        Sempre que possível formate a resposta em listas, tabelas ou exemplos para melhor compreensão.
        
        Contexto:
        {contexto}

        Histórico da conversa:
        {historico}

        Pergunta: {entrada}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("user", "{entrada}")
    ])

    llm = ChatGithubCopilot(
        temperature=0.7,
        model="gpt-4o-mini",
        github_token=copilot_token,
    )

    return prompt | llm
```

#### O que acontece aqui?

- `template` define as instruções do sistema.
- `ChatPromptTemplate.from_messages(...)` monta o prompt em formato estruturado com duas partes:
  - `system`: instruções gerais
  - `user`: pergunta do usuário
- `ChatGithubCopilot(...)` cria o cliente do modelo do GitHub Copilot.
- `return prompt | llm` conecta prompt ao modelo, criando uma cadeia LangChain.

#### Parâmetros importantes:

- `temperature=0.7`
  - controla a criatividade da resposta
  - valores mais altos geram respostas mais variadas

- `model="gpt-4o-mini"`
  - define qual modelo será usado

- `github_token=copilot_token`
  - permite autenticação do modelo no serviço do Copilot

#### Pacotes externos usados

- `langchain_core.prompts.ChatPromptTemplate`
  - cria templates de mensagem para LLMs

- `langchain_githubcopilot_chat.ChatGithubCopilot`
  - conecta o LangChain ao modelo do GitHub Copilot

---

### 5. Loop da conversa

Arquivo: `src/conversa.py`

```python
from autenticacao import obter_token
from chain import criar_chain
from rag import obter_rag


def iniciar_conversa():
    copilot_token = obter_token()
    retriever = obter_rag()
    chain = criar_chain(copilot_token)

    print("Pergunte o que quiser sobre tributação de fundos de investimento. (Digite '/quit' para encerrar)")
    historico = []

    while True:
        pergunta_usuario = input("\nVocê: ").strip()
        if not pergunta_usuario:
            continue
        if pergunta_usuario.lower() == "/quit":
            print("Encerrando conversa...")
            break

        documentos_relevantes = retriever.invoke(pergunta_usuario)
        contexto = "\n\n".join(
            doc.page_content for doc in documentos_relevantes
        )

        historico_formatado = "\n".join(
            f"Usuário: {pergunta}\nAssistente: {resposta}"
            for pergunta, resposta in historico
        )
        resposta = chain.invoke({
            "entrada": pergunta_usuario,
            "contexto": contexto,
            "historico": historico_formatado,
        })
        print(f"Assistente: {resposta.content}")
        historico.append((pergunta_usuario, resposta.content))
```

#### O que acontece aqui?

- `obter_token()` autentica o Copilot
- `obter_rag()` prepara a base de conhecimento
- `criar_chain(...)` monta a cadeia do modelo
- o script entra em um laço infinito esperando perguntas do usuário

Para cada pergunta:

1. `retriever.invoke(pergunta_usuario)` busca os documentos mais relevantes de `rag.md`
2. você monta o `contexto` com os trechos relevantes
3. o histórico da conversa é convertido em texto
4. chama `chain.invoke(...)` com:
   - pergunta atual
   - contexto recuperado
   - histórico anterior
5. imprime a resposta
6. salva no histórico para melhorar a continuidade da conversa

#### Como o contexto e o histórico ajudam?

- `contexto`: fornece fatos do material de referência
- `historico`: permite que a conversa seja coerente
- `prompt`: orienta o modelo a responder apenas com base no contexto e no tema

---

## Fluxo completo da aplicação

```text
usuário pergunta
   ↓
obter_token()
   ↓
obter_rag()
   ↓
criar_chain()
   ↓
retriever.invoke(pergunta)
   ↓
contexto + histórico + prompt
   ↓
modelo GitHub Copilot
   ↓
resposta ao usuário
```

---

## Conceito geral: RAG

RAG significa Retrieval-Augmented Generation, ou seja:

- o sistema recupera informações relevantes de uma base de conhecimento;
- então o modelo usa essas informações para gerar uma resposta mais precisa e contextualizada.

No projeto:

- `rag.md` é a base de conhecimento;
- `BM25Retriever` faz a recuperação;
- `GitHub Copilot` gera a resposta final.

---

## Observações importantes

- O sistema só responde sobre tributação de fundos de investimento.
- Qualquer pergunta fora do tema é ignorada pela instrução do sistema.
- Se a resposta não estiver no contexto, o modelo é instruído a dizer que a informação não consta no material disponível.
- O histórico é usado para manter continuidade, mas a resposta principal continua baseada no contexto relevante.

---

## Como executar

1. Instale as dependências:

```bash
pip install -r config/requirements.txt
```

2. Certifique-se de que o arquivo `rag.md` exista e contenha o material de referência.

3. Execute:

```bash
python app.py
```

4. Digite uma pergunta e pressione Enter.

5. Para encerrar, digite:

```text
/quit
```

---

## Dependências principais

O projeto usa principalmente:

- `python-dotenv`
- `langchain`
- `langchain-community`
- `langchain-core`
- `langchain-text-splitters`
- `langchain-githubcopilot-chat`

Essas bibliotecas permitem:

- leitura de arquivos e dados;
- construção de prompts;
- busca por relevância;
- autenticação e integrações com modelos de IA.

---

## Conclusão

Esse projeto combina busca e geração para criar um assistente que responde com base em um material específico. A parte principal é:

- recuperar o texto certo do documento;
- enviar esse contexto para o modelo;
- usar instruções claras para controlar o comportamento e a qualidade da resposta.

O resultado é uma IA mais focada, útil e consistente com o tema do material de referência.
