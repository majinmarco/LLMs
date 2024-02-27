# What is this?

This chatbot is one that utilizes a concept called Retrieval Augmented Generation. You can provide PDF and text files in the docs folder. These are then converted to a vector database AKA a knowledge base. When you provide a prompt to the AI, it takes that prompt and plugs it into a prompt template, while at the same time using it in a simialrity search function for the knowledge base which finds information that could help the AI answer your question. It plugs this information in, as well as the previously mentioned prompt, into the prompt template. Through this input, the AI takes the question and context that was puleld from the knowledge base in order to answer your question.

# How to use

1. Download the repo.
2. Make sure you have all necessary packages installed, create an LLM environnebt if needed, especially if you want to try out future projects like this one.
3. Everything should be set up after this, simply open a terminal/cmd prompt in the folder and type `streamlit run rag.py`
4. A window will automatically pop up with the chatbot.

The chatbot is designed to understand questions in both English and Spanish, thanks to the multilingual embeddings used, allowing for a wider range of interactions.
