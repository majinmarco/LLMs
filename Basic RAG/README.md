# What is this?

This chatbot is one that utilizes a concept called Retrieval Augmented Generation. You can provide PDF and text files in the docs folder. These are then converted to a vector database AKA a knowledge base. When you provide a prompt to the AI, it takes that prompt and plugs it into a prompt template, while at the same time using it in a simialrity search function for the knowledge base which finds information that could help the AI answer your question. It plugs this information in, as well as the previously mentioned prompt, into the prompt template. Through this input, the AI takes the question and context that was puleld from the knowledge base in order to answer your question.

# How to use

1. Download the repo.
2. Make sure you have all necessary packages installed, create an LLM environemtn if needed, especially if you want to try out future projects like this one.
3. Everything should be set up after this, simply open a terminal/cmd prompt in the folder and type `streamlit run rag.py`
4. A window will automatically pop up with the chatbot.

# How to use

After setting up your development environment as described in the previous sections, you can easily start using the chatbot by following these steps:

1. Open a terminal or command prompt window in the folder where you cloned the repository.

2. Run the chatbot application by executing the following command:

   ```
   poetry run streamlit run rag.py
   ```

3. A window will automatically open in your default web browser with the chatbot interface. Here, you can interact with the chatbot by typing your questions into the provided text box.

The chatbot is designed to understand questions in both English and Spanish, thanks to the multilingual embeddings used, allowing for a wider range of interactions.