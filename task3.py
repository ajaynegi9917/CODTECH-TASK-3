import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK setup (fixes punkt_tab issue) ---
for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
    try:
        if pkg in ["punkt", "punkt_tab"]:
            nltk.data.find(f"tokenizers/{pkg}")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# --- Q/A pairs ---
qa_pairs = {
    "What is NLP?": "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language.",
    "What is NLTK?": "NLTK (Natural Language Toolkit) is a powerful Python library for working with human language data.",
    "What is a chatbot?": "A chatbot is a software application used to conduct an online chat conversation via text or text-to-speech, instead of direct contact with a human agent.",
    "How does a chatbot work?": "Chatbots work by analyzing the user's input, identifying the intent, and providing a predefined or generated response.",
    "What are the types of chatbots?": "Chatbots can be classified into two main types: rule-based and AI-powered (generative or retrieval-based).",
    "Thank you": "You're welcome! Is there anything else I can help you with?",
    "Hi": "Hello! How can I help you today?",
    "Hello": "Hi there! How may I assist you?"
}

# --- Preprocessing helpers ---
def normalize(text: str) -> str:
    """Lowercase + remove punctuation + strip spaces"""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

lemmatizer = nltk.WordNetLemmatizer()

def tokenize_and_lemmatize(text: str):
    """Tokenizer for TF-IDF: lower, remove punctuation, tokenize, lemmatize"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(tok) for tok in tokens]

# --- Prepare TF-IDF ---
questions = list(qa_pairs.keys())
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_and_lemmatize,
    stop_words='english',
    token_pattern=None  # needed when using custom tokenizer
)
tfidf_matrix = vectorizer.fit_transform(questions)

# --- Chatbot response ---
def chatbot_response(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return "Please type something so I can help."

    norm_in = normalize(user_input)

    # Exit conditions
    if norm_in in {"bye", "exit", "quit"}:
        return "Goodbye! Have a great day."

    # Direct match (case-insensitive)
    normalized_questions = [normalize(q) for q in questions]
    if norm_in in normalized_questions:
        idx = normalized_questions.index(norm_in)
        return qa_pairs[questions[idx]]

    # TF-IDF similarity
    user_tfidf = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    threshold = 0.30
    if best_score > threshold:
        return qa_pairs[questions[best_idx]]

    # Suggest close matches if confidence is low
    top_n = similarities.argsort()[-3:][::-1]
    suggestions = [questions[i] for i in top_n if similarities[i] > 0]
    if suggestions:
        suggestion_text = "\n- ".join(suggestions)
        return ("I'm not sure I understood. Did you mean:\n- " + suggestion_text)

    return "I'm sorry, I don't have an answer for that. Can you please rephrase?"

# --- Run chatbot ---
if __name__ == "__main__":
    print("Chatbot: Hello! I'm here to answer your questions about NLP and chatbots. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        reply = chatbot_response(user_input)
        print(f"Chatbot: {reply}")
        if reply == "Goodbye! Have a great day.":
            break
