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