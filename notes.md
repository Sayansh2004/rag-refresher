## How RAG system works?

-> It works in 3 phases;

# 1.Document ingestion phase
 preprocessing happens in this phase i.e. the data is converted into smaller chunks using document splitter and those chunks are then fed to embedding model and those embedding are then fed to vector database.

 chunking is done due the presence of context window as an LLM can take a data to a certain range till which it remembers the context and this is known as context window.

 so we use similarity search such as cosine similarities to find the similarity of user query in the vector store.


 Data->Document splitter->chunks->embedding(vectors)->vector store.


 # 2. Query Processing Phase

 In this process the user query gets converted to embedding using the same embedding model which is used to generate embeddings of the data.Then a similarity search is performed on these embeddings

 user query(Input)->Embedding Model->vectors(with some dimensions)->search the relevant text in the vector database->based on similarity search we get chunks of documents->context->which is given to LLM.

 # 3. Generation Phase

 In this phase the context which came from query processing phase,then the metadata is added which is called augmentation.


 original Query + enriched context->LLMs (openAI,LLAMA,Gemini)-> Summarized output.