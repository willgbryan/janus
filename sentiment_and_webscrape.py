import requests
from bs4 import BeautifulSoup
import openai

# Initialize OpenAI API with your key
openai.api_key = 'redacted'

# Function to scrape headlines
def scrape_headlines(company):
    url = f'https://www.google.com/search?q={company}+news&tbm=nws'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    headlines = []
    for item in soup.find_all('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}):
        headlines.append(item.get_text())
    return headlines

# Function to analyze sentiment of a headline
def analyze_sentiment(headline):
    response = openai.Completion.create(
      engine="text-davinci-004",
      prompt=headline + "\nThe sentiment of the above text is:",
      temperature=0.3,
      max_tokens=60
    )
    return response.choices[0].text.strip()

# Example usage
company = 'AAPL'
headlines = scrape_headlines(company)
print(headlines)

sentiments = []
for headline in headlines:
    sentiment = analyze_sentiment(headline)
    print(f'Headline: {headline}\nSentiment: {sentiment}\n')
    sentiments.append(sentiment)
