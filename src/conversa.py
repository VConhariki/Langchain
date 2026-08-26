from src.autenticacao import obter_token
from src.chain import criar_chain
from src.rag import obter_rag

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