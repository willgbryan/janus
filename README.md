# janus
Transformer based forecasting platform built on PyTorch Forecasting and PyTorch Lightning: 

Current build is heavily configured for market forecasting and creates technical analysis features for RSI, MACD, Bollinger Bands, a couple of different moving averages, and an assortment of date features. Future builds will attempt to ease user feature selection as the current process for adding or removing features requires modifications to the config, data_setup method, and predict method.

It's worth noting that the config helps to consolidate what otherwise would be a slew of hardcoded values throughout the class. It doesn't do a good job of acting as a true config.

Early results:



![meta](https://user-images.githubusercontent.com/107731540/236980035-1ec22992-b285-4909-97a9-fc0c7d24dd09.png)
![msft](https://user-images.githubusercontent.com/107731540/236980023-a4d1e321-3fbb-4ed1-8d22-5c17181302ff.png)
![nvda](https://user-images.githubusercontent.com/107731540/236980024-4d530641-e356-4966-9877-b3f7f553f720.png)
![tsla](https://user-images.githubusercontent.com/107731540/236980025-9cac9b95-8934-46ea-a2d3-facccfb99554.png)
![aapl](https://user-images.githubusercontent.com/107731540/236980027-dd3a545e-78a0-4d1a-a247-9645f5dc395b.png)
![amd](https://user-images.githubusercontent.com/107731540/236980029-f794595a-3a8f-46d6-8695-3a881470a944.png)
![amzn](https://user-images.githubusercontent.com/107731540/236980030-0b4a4ee3-54d6-4c0e-b0fd-c988f1279b32.png)
![crm](https://user-images.githubusercontent.com/107731540/236980031-70762d5f-6e05-4715-81bd-1402fc17ac75.png)
![googl](https://user-images.githubusercontent.com/107731540/236980032-3dcb0c54-97d7-4704-9d11-4f485c7dbd26.png)
![intc](https://user-images.githubusercontent.com/107731540/236980034-abd1cbb1-1c5f-4f54-9f54-c584cf5d92d5.png)
