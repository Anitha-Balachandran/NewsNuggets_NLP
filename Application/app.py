from flask import Flask, request, render_template
import pickle
import google.generativeai as genai
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure required NLTK resources are downloaded
nltk.download("stopwords")
nltk.download("wordnet")


# Define the TextPreprocessor class
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def remove_tags(self, text):
        """Remove HTML tags."""
        return re.sub(r"<.*?>", "", text)

    def remove_special(self, text):
        """Remove special characters."""
        return text.translate(str.maketrans("", "", string.punctuation))

    def convert_to_lower(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def lemmatize_text(self, text):
        """Lemmatize the text."""
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def remove_stopwords(self, text):
        """Remove stopwords from the text."""
        return " ".join(
            [word for word in text.split() if word.lower() not in self.stop_words]
        )

    def preprocess(self, text):
        """Apply all preprocessing steps to the text data."""
        text = self.remove_tags(text)
        text = self.remove_special(text)
        text = self.convert_to_lower(text)
        text = self.lemmatize_text(text)
        text = self.remove_stopwords(text)
        return text


# Summarizer class
class NewsSummarizer:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_summary(self, article_content):
        config = genai.types.GenerationConfig(
            temperature=0.5,
            max_output_tokens=200,
            top_k=40,
            top_p=0.9,
            stop_sequences=["\n\n"],
        )
        response = self.model.generate_content(
            f"Summarize the following news article in 200 words:\n\n{article_content}\n\n",
            generation_config=config,
        )
        return response.text


# Initialize Flask app
app = Flask(__name__)

# Load the pickled preprocessor, vectorizer, and classification model
with open("news_preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("multinomial_nb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the summarizer
api_key = "AIzaSyB1yxyvqjVa6gqvVi9I23UKkwxHhMVcetY"
summarizer = NewsSummarizer(api_key)

# Mapping of original categories to encoded values
category_mapping = {
    0: "business",
    1: "entertainment",
    2: "politics",
    3: "sport",
    4: "tech",
}


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    text = request.form["text"]
    preprocessor = TextPreprocessor()

    # Preprocess the text
    preprocessed_text = preprocessor.preprocess(text)

    # Vectorize the preprocessed text
    X_test = vectorizer.transform([preprocessed_text])

    # Predict the category
    category_id = model.predict(X_test)[0]
    category = category_mapping[category_id]

    # Generate a summary of the article
    summary = summarizer.generate_summary(text)

    return render_template("result.html", category=category, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
