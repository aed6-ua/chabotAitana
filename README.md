# Comandos:

- Iniciar container:  docker-compose up -d
- Ejecutar en el container: 
    - docker exec -it rag python rag.py argumentos
    - docker exec -it rag python server.py
    - (parametre -d per fer-lo en background)
- Parar container: docker stop rag
- Vore processos dins el container: docker top rag

- OLLAMA: (esta en un container anomenat 'ollama')
    - docker exec ollama comando
    - Exemple: docker exec ollama ollama list --> per vore els models carregats

# TODO:

- Parametrizar todas las constantes y datos posibles en un env.
- indexManager dentro de retriever
- vore codi Eduard: model, etc.
- Embeddings:
    - SentenceTransformer
    - LlamaIndex
- Retrievers:
    - SentenceTransformer
    - LlamaIndex
- Assistants
    - GPT
    - LlamaIndex
    - Local LLM
- busqueda bilingüe, en función de un argumento en el post
- ver de poder tener varios asistentes que sean invocados con un argumento en el post (o en distintos puertos)
- Usar llamaIndex solo para búsqueda semántica y la ingesta en varios formatos
- 'Atacar' al LLM directamente (que se pueda escoger a qué LLM)
- crear LocalLLMAssistant
- Tema multicontexto

# DONE

- retrieverstrategy, poner los métodos abstractos de retriever, hacer lo mismo con embeddings (o similar)
- transformar el env en json o integrar en el config.json
- unificar en un fichero el retriver, assistant y embeddings