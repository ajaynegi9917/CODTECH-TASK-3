# CODTECH-TASK-3

COMPANY : CODTECH IT SOLUTIONS

NAME : AJAY NEGI

INTERN ID : CT08DY1001

DOMAIN : PYTHON PROGRAMMING

DURATION : 8 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

This project is a retrieval-based chatbot implemented in Python using NLTK (Natural Language Toolkit) and Scikit-learn. The chatbot is designed to answer questions about Natural Language Processing (NLP), NLTK, and chatbots. Unlike generative AI models, this chatbot relies on a predefined set of question-answer pairs and uses traditional information retrieval techniques to provide relevant responses. The main goal of this project is to demonstrate how NLP techniques and vector space models can be combined to build a functional, lightweight chatbot suitable for small domains.

The chatbot begins by preprocessing user input and stored questions to ensure accurate matching. Preprocessing involves converting text to lowercase, removing punctuation, and lemmatizing words using NLTK’s WordNet lemmatizer. Lemmatization reduces words to their base forms, which improves the ability to match similar words such as “running” and “run.” This normalized text is then transformed into numerical vectors using TF-IDF (Term Frequency–Inverse Document Frequency). TF-IDF assigns weights to words based on their frequency across all stored questions, helping the system distinguish between common words and those that carry more meaning in the context of the knowledge base.

Once user input is vectorized, the chatbot computes cosine similarity between the input vector and all stored question vectors. Cosine similarity measures how closely the user query aligns with each predefined question. If the similarity score exceeds a predefined threshold (0.3 in this implementation), the chatbot returns the corresponding answer. This approach ensures that the chatbot provides accurate responses while ignoring irrelevant inputs. If no strong match is found, the chatbot suggests up to three similar questions to help the user rephrase their query. Additionally, direct case-insensitive matches are checked first to provide instant responses for exact queries such as “What is NLP?” or “Hi.”

The chatbot also includes handling for exit commands, including “exit,” “quit,” and “bye,” allowing users to end the conversation gracefully. Special responses for greetings and expressions of thanks improve the user experience, making interactions feel more natural and conversational.

This project highlights several key NLP concepts, including text normalization, tokenization, lemmatization, vectorization, and similarity-based retrieval. It demonstrates how traditional NLP techniques can be applied to create intelligent systems without relying on large-scale neural networks or pretrained language models. The code is modular, with separate functions for normalization, tokenization, vectorization, and response selection, making it easy to extend. New question-answer pairs can be added to expand the knowledge base, and thresholds can be adjusted to improve response sensitivity.

In conclusion, this chatbot is a practical example of applying classical NLP methods to build a functional conversational agent. It is lightweight, fast, and easy to maintain, providing a strong foundation for learners interested in chatbot development, text mining, and natural language understanding. This project can be extended with more sophisticated features, such as conversation memory, multi-turn dialogue handling, or integration with web or messaging platforms, making it a versatile starting point for more advanced NLP applications.

#OUTPUT

<img width="960" height="540" alt="Image" src="https://github.com/user-attachments/assets/d41be412-f12b-4eef-8e1d-e6bf168caa4e" />

