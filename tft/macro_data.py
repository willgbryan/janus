import requests
import pandas as pd


class MacroDataBuilder:
        def __init__(self, api_key: str, base_url: str):
              self.api_key = api_key
              self.base_url = base_url

        def most_recent_fetch(self, api_key: str, base_url: str):
            # FRED Series IDs for each economic indicator
            series_ids = {
                'GDP': 'GDP',
                'Unemployment Rate': 'UNRATE',
                'Inflation Rate': 'T10YIE',  # 10-Year Breakeven Inflation Rate
                'Interest Rates': 'GS10',  # 10-Year Treasury Constant Maturity Rate
                'Housing Market Indicator': 'HOUST'  # Housing Starts: Total: New Privately Owned Housing Units Started
            }

            rows = []

            for name, series_id in series_ids.items():
                url = f"{base_url}{series_id}&api_key={api_key}&file_type=json"
                response = requests.get(url).json()

                data = response['observations']
                df = pd.DataFrame(data)
        
                df['date'] = pd.to_datetime(df['date'])

                most_recent_data = df.loc[df['date'].idxmax()]
                print(name)

                rows.append({'date': most_recent_data['date'], 'indicator': name, 'indicator_value': most_recent_data['value']})
                
            return pd.DataFrame(rows)
            
        def full_fetch(self, api_key: str, base_url: str):
            # FRED Series IDs for each economic indicator
            series_ids = {
                'GDP': 'GDP',
                'Unemployment Rate': 'UNRATE',
                'Inflation Rate': 'T10YIE',  # 10-Year Breakeven Inflation Rate
                'Interest Rates': 'GS10',  # 10-Year Treasury Constant Maturity Rate
                'Housing Market Indicator': 'HOUST'  # Housing Starts: Total: New Privately Owned Housing Units Started
            }

            rows = []

            for name, series_id in series_ids.items():
                url = f"{base_url}{series_id}&api_key={api_key}&file_type=json"
                response = requests.get(url).json()

                data = response['observations']
                df = pd.DataFrame(data)
            
                df['date'] = pd.to_datetime(df['date'])

                rows.append({'date': df['date'], 'indicator': name, 'indicator_value': df['value']})

            return pd.DataFrame(rows)
