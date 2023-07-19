import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
from macro_data import MacroDataBuilder

macro_information = MacroDataBuilder(api_key='redacted', base_url='https://api.stlouisfed.org/fred/series/observations?series_id=')
current_date = '2023-05-20'

# Function to set up OpenAI API
def setup_openai(api_key):
    session = requests.Session()
    session.headers.update({
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    })
    return session

# Function to create chat completion
def create_chat_completion(session, messages, options=None):
    if options is None:
        options = {}
    model = options.get('model', 'gpt-3.5-turbo')
    data = {
        'model': model,
        'messages': messages,
        'temperature': options.get('temperature', 0.0)
    }
    data.update(options)
    response = session.post('https://api.openai.com/v1/chat/completions', data=json.dumps(data))

    # Error handling
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        print(f'Error making API request: {error}')

    return response.json().get('choices')

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
def analyze_sentiment(session, headline):
    messages = [
        {'role': 'system', 'content': 'Forget all of your previous instructions. Pretend you are a financial expert. You are a financial expert with stock recommendation experience. Answer on a scale of 1-10, 10 being good news and 1 being bad news, or "UNKNOWN" if uncertain on the first line. You must return ONLY a sentiment score and nothing else.'},
        {'role': 'user', 'content': headline},
    ]
    choices = create_chat_completion(session, messages)
    if choices:
        return choices[0]['message']['content']
    else:
        return None
    
def parse_list(sentiments):
    new_list = []
    for item in sentiments:
        try:
            new_list.append(float(item))
        except (ValueError, TypeError):
            continue
    return new_list



openai = setup_openai('redacted')

ticker_list_tech = [
    "AMZN", 
    "AAPL", 
    "GOOGL", 
    "MSFT", 
    "NVDA", 
    "META", 
    "TSLA", 
    "AMD", 
    "INTC", 
    "CRM",    
    "SOFI",
    "PYPL",
    "SNOW",
    "ADBE",
    "UBER",
    "ABNB",
    "FUBO",
    "AVGO",
    "ORCL",
    "ASML",
    #"CSCO",
    "ACN",
]

ticker_list_energy = [
    "XOM",
    "CVX",
    "SHEL",
    "TTE",
    "COP",
    "BP",
    "EQNR",
    #"ENB",
]


ticker_list_market_index = [
    "^NDX", 
    "^GSPC", 
    "^DJI",
    "^RUT",
    "^IXIC",
    
]

ticker_list_financial_services = [
    "BRK-A",
    "BRK-B",
    "V",
    "JPM",
    "MA",
    "BAC",
    "WFC",
    "MS",
    "HSBC",
]

ticker_lists = {
    "tech":ticker_list_tech, 
    "index":ticker_list_market_index, 
    "financial":ticker_list_financial_services,
    "energy":ticker_list_energy,
}
rows = []

for key, value in ticker_lists.items():
    for i in value:
        company = i
        headlines = scrape_headlines(company)

        sentiments = []
        for i in headlines:
            headline = f"The sentiment for the following news headline is what: {i}."
            sentiment = analyze_sentiment(openai, headline)
            print(f'Headline: {headline}\nSentiment: {sentiment}\n')
            sentiments.append(sentiment)

        cleaned_sentiments = parse_list(sentiments=sentiments)
        rows.append({'ticker': company, 'sentiment_score': sum(cleaned_sentiments) / len(cleaned_sentiments) })


sentiment_dataframe = pd.DataFrame(rows)
sentiment_dataframe['date'] = current_date
print(sentiment_dataframe)

sentiment_dataframe.to_csv('sentiment_dataframe.csv', index=False)


