import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Porter Stemmer
ps = PorterStemmer()

# Load vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Filter alphanumeric words
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Streamlit app with custom CSS
def main():
    # Custom CSS for background color
    bg_color = """
        <style>
        body {
            background-color: #dcdcdc; /* Light gray background color */
        }
        </style>
    """
    st.markdown(bg_color, unsafe_allow_html=True)  # Inject custom CSS for background color

    st.title("Email Spam Classifier")
    st.write("Enter a message to determine if it's spam or not.")

    input_sms = st.text_area("Input Message")

    if st.button('Predict'):
        if input_sms.strip():  # Check if input is not empty
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.error("This message is classified as **Spam**.")
            else:
                st.success("This message is classified as **Not Spam**.")
        else:
            st.warning("⚠️ Please enter a message.")

if __name__ == '__main__':
    main()
