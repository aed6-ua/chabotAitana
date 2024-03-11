# Comandos:

- Iniciar container:  docker-compose up -d
- Ejecutar en el container: docker exec rag python p.py argumentos
- Parar container: docker stop rag

# TODO:

- Parametrizar todas las constantes y datos posibles en un env.
- index manager dentro de retriever
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
