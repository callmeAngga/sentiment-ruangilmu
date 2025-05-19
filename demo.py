import joblib

def predict_sentiment(text):
    model = joblib.load('my-model/model.joblib')
    prediction = model.predict([text])[0]

    return prediction


# Contoh penggunaan
if __name__ == "__main__":
    text = "Bagus, penjelasan sangat jelas dan mudah dipahami"
    sentiment = predict_sentiment(text)
    if sentiment == 1:
        sentiment = "Negatif"
    elif sentiment == 2:
        sentiment = "Positif"
        
    print(f"Input   : {text}")
    print(f"Output  : {sentiment}")
