"""
Module to analyse a given dataset


Add logging and unit tests
"""

#Imports
import pandas as pd
import numpy as np
import logging as lg
import matplotlib.pyplot as plt
import arch
import time
import math



#Logging config
lg.basicConfig(level=lg.WARNING,format='%(process)d-%(levelname)s-%(message)s')


class CSV:
   
    def __init__(self,path: str):
        self._path = path
        self._symbolinfo = []
        self._symboldata = 0
    
    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str) -> str:
        self._path = value
    
    def symboldata(self) -> None:
        return self._symboldata

    def symbolinfo(self) -> None:
        return self._symbolinfo

    def csv_to_pd(self) -> pd.DataFrame:
        lg.info("Converting CSV to pandas Dataframe")
        return pd.read_csv(self.path)

    def data_view(self,dataframe: pd.DataFrame) -> None:
        data = dataframe
        lg.info("Adding columns names")
        data.columns = ["Timestamp","Symbol","Volume","Price"]
        print(data.info())
        print(f"Head: \n{data.head()}\n Tail: {data.tail()}")

    def data_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        #function for feature extraction
        lg.info("Dropping all N/A values from the dataframe")
        df.dropna(inplace=True)
        lg.info("Adding column names")
        df.columns = ["Timestamp","Symbol","Volume","Price"]
        # df = df.get_dummies(df,column) - use if needed
        return df 

    def data_symbol_extraction(self, df: pd.DataFrame) -> None:
        symbols = list(df["Symbol"].unique())
        lg.info("Grouping by symbol")
        grouped_data = df.groupby("Symbol")
        return grouped_data

    def feature_extraction(self, grouped_data: pd.DataFrame) -> pd.DataFrame:
        data_multi = {}
        for symbol, attributes in grouped_data:
            data_multi[symbol] = attributes
        try:
            lg.info("Calculating metrics")
            vol_cum = [data_multi[k]["Volume"].sum() for k in data_multi]
            minPrice = [data_multi[k]["Price"].min() for k in data_multi]
            maxPrice = [data_multi[k]["Price"].max() for k in data_multi]
            averagePrice = [data_multi[k]["Price"].mean() for k in data_multi]
        except:
            lg.error("Invalid types in data")

        symbols = []
        for i in range(len(data_multi)):
            symbols.append([vol_cum[i],minPrice[i],maxPrice[i] 
                            ,averagePrice[i]])
       
        self._symboldata = data_multi
        self._symbolinfo = symbols
        return symbols, data_multi

    def run(self):
        self.feature_extraction(self.data_symbol_extraction(
            self.data_preprocess(self.csv_to_pd())))






class Instrument:
    
    def __init__(self, name, data):
        self._name = name
        self._maxPrice = 0
        self._minPrice = 0
        self._averagePrice = 0
        self._tradeCount = 0
        self._totalVolume = 0
        self._data = data


    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> str:
        self._name = value

    @property
    def maxPrice(self) -> int:
        return self._maxPrice

    @maxPrice.setter
    def maxPrice(self, value: int) -> int:
        self._maxPrice = value

    @property
    def minPrice(self) -> int:
        return self._minPrice

    @minPrice.setter
    def minPrice(self, value: int) -> int:
        self._minPrice = value

    @property
    def averagePrice(self) -> int:
        return self._averagePrice

    @averagePrice.setter
    def averagePrice(self, value: int) -> int:
        self._averagePrice = value

    @property
    def tradeCount(self) -> int:
        return self._tradeCount

    @tradeCount.setter
    def tradeCount(self, value: int) -> int:
        self._tradeCount = value

    @property
    def totalVolume(self) -> int:
        return self._totalVolume

    @totalVolume.setter
    def totalVolume(self, value) -> int:
        self._totalVolume = value

    @property
    def data(self) -> int:
        return self._data
    
    
    def addTrade(self, timestamp: int, volume: int, price: int) -> None:
        data_to_append = [ {
        "Timestamp": timestamp,
        "Symbol": self.name,
        "Volume": volume,
        "Price": price
        }]
        df_to_append = pd.DataFrame(data_to_append)
        self._data = self._data._append(df_to_append, ignore_index=True)
        lg.info("New trade added")

    def printSummary(self) -> None:

        print(f'Symbol: {self.name} Max Price: {self.maxPrice} Min Price: {self.minPrice} Average Price: {self.averagePrice} Total Volume: {self.totalVolume}')
        

    def garman_klass_vol(self) -> None:
        """
        Garman-Klass Volatility = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} \left(\frac{1}{2} \cdot 
                                \left( \log{\frac{H_i}{L_i}} \right)^2 - \frac{2\log{C_i}}{H_i + L_i}\right)}
        n is the number of data points
        H_i,L_i  are the highest and lowest prices in a given time period 
        C_i is the closing price in the time period

        """
        data = self.data
        open_price = data.iloc[0, 3]
        close_price = data.iloc[-1,3]
        num_rows, num_columns = data.shape
        try:
            ln_ratio_open_close = math.log(close_price / open_price)
            ln_ratio_high_low = math.log(self.maxPrice / self.minPrice)
            
            first_component = (0.5 * ln_ratio_open_close) ** 2
            second_component = (2 * ln_ratio_high_low - ln_ratio_open_close) ** 2
            
            average_daily_volatility = (first_component - second_component) / (num_rows * math.log(2))
            print(f"Average Daily Volatility: {average_daily_volatility}")
            realised_volatility = math.sqrt(abs(average_daily_volatility))
             
            return realised_volatility
        except:
            lg.error("Math Error")



    def garch(self) -> None:
        """
        \sigma_t^2 = \omega + \alpha \cdot \varepsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2
        
         σ{t}^2 represents the conditional variance at time t
        \omega is the constant term or intercept of the GARCH model
        α and β are the coefficients that measure the impact of the lagged squared error term 
        ε_{t−1}^2 and the lagged conditional variance σ_{t-1}^2 on the current conditional variance
        σ_t^2, respectively.

        """

        model = arch.arch_model(self.data['Price'].dropna(), mean='Zero', vol='Garch', p=2, q=2)

        # Estimate the model parameters
        model_fit = model.fit()

        # Forecast volatility for a specific number of steps ahead
        forecast_horizon = 5
        forecast = model_fit.forecast(horizon=forecast_horizon)

        # Get the forecasted volatility for the last observation
        forecasted_volatility = forecast.variance[-1:].values[0]

        print("Model Summary:")
        print(model_fit.summary())
        print("\nForecasted Volatility:")
        print(forecasted_volatility)

    def ewma(self) -> None:
        """
        EWMA(\alpha)_t = \alpha \cdot \text{data}_t + (1 - \alpha) \cdot \text{EWMA}(\alpha)_{t-1}
        \alpha is the smoothing factor or decay factor, which lies between 0 and 1. 
        It determines how much weight to assign to the current data point relative to the previous EWMA value

        """
        data = self.data
        alpha = 0.2

        returns = data['Price'].pct_change()
    
        # Calculate squared returns and apply EWMA
        squared_returns = returns ** 2
        ewma_squared_returns = squared_returns.ewm(alpha=alpha, adjust=False).mean()
    
        # Calculate EWMA volatility as the square root of the smoothed squared returns
        ewma_volatility = ewma_squared_returns ** 0.5
        plt.plot(ewma_volatility)
        plt.title("EWMA Volatility")
        plt.xlabel("Time")
        plt.ylabel("Volatility")
        plt.show()
    



    def plot(self) -> None:
        data = self.data
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the 'Price' data
        ax.plot(data['Timestamp'], data['Price'], label='Price', color='b')
        # Set labels and title
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price')
        ax.set_title('Price over Time')
        # Rotate the x-axis labels for better visibility
        plt.xticks(rotation=45)
        # Show the legend
        ax.legend()
        # Display the plot
        plt.show()
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the 'Volume' data
        ax.plot(data['Timestamp'], data['Volume'], label='Volume', color='g')
        # Set labels and title
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Volume')
        ax.set_title('Volume over Time')
        # Rotate the x-axis labels for better visibility
        plt.xticks(rotation=45)
        # Show the legend
        ax.legend()
        # Display the plot
        plt.show()
        pass



def addInstruments(path: str) -> dict[str,pd.DataFrame]:
    
    preprocess = CSV(path)
    preprocess.run()
    symbols, data = preprocess.symbolinfo(), preprocess.symboldata()
    allsymbols = {}
    j = 0
    for symbol in data:
        instru = Instrument(symbol, data[symbol])
        lg.info(f"New Instrument {instru.name} added")
        instru.totalVolume =symbols[j][0]
        instru.minPrice = symbols[j][1]
        instru.maxPrice = symbols[j][2] 
        instru.averagePrice = symbols[j][3]
        j+=1 
        allsymbols[symbol] = instru
    
    

    return allsymbols




def accessInstrument(symbol: str,symbolsdata: dict[str, pd.DataFrame], choice: int) -> None:
    
    if choice == 1:
        symbolsdata[symbol].printSummary()
    elif choice == 2:
        print(f"Enter trade for {symbol}")
        timestamp = int(input("Enter timestamp: "))
        volume = int(input("Enter volume: "))
        price = int(input("Enter price: "))
        symbolsdata[symbol].addTrade(timestamp, volume, price)
    elif choice == 3:
        symbolsdata[symbol].garch()
    elif choice == 4:
        symbolsdata[symbol].plot()
    elif choice == 5: 
        symbolsdata[symbol].ewma()
    else: 
        print(f"Realised Volatility: {symbolsdata[symbol].garman_klass_vol()}")




def show_menu():
    print("\nMenu:")
    print("1. Show Summary")
    print("2. Add Trade")
    print("3. Use GARCH")
    print("4. Plot Instrument Data")
    print("5. Plot EWMA Volatility")
    print("6. Use Garman-Klass")
    print("0. Exit")

def bulk_take():
    symbols = input("Enter the symbols in the format aaa,aab,aac ... \n").lower()
    symbols = symbols.replace(" ", "")
    symbols_list = list(symbols.split(","))
    return symbols_list


def menu():
    instruments = addInstruments("input_data.csv")
    symbols = []
    while True:
        print("-----------------------------------")
        print("Symbol Entry:")
        bulk = input("For single/bulk entry enter 0, for ALL enter 1: ")
        if bulk != "1":
            symbols = bulk_take()
        else:
            symbols = list(instruments.keys())
        print("-----------------------------------")
        show_menu()
        choice = input("Enter Choice: ") 
        print("-----------------------------------")
        if choice == '1':
            for symbol in symbols:
                try:
                    accessInstrument(symbol,instruments,1)
                except:
                    lg.error(f"INVALID-SYMBOL-{symbol}")
            time.sleep(2)
        elif choice == '2':
            for symbol in symbols:
                try:
                    accessInstrument(symbol,instruments,2)
                except:
                    lg.error(f"INVALID-SYMBOL-{symbol}")
            time.sleep(2)
        elif choice == '3':
            for symbol in symbols:
                try:
                    accessInstrument(symbol,instruments,3)
                except:
                    lg.error(f"INVALID-SYMBOL-{symbol}")
            time.sleep(2)
        elif choice == '4':
            for symbol in symbols:
                try:
                    accessInstrument(symbol,instruments,4)
                except:
                    lg.error(f"INVALID-SYMBOL-{symbol}")
            time.sleep(2)
        elif choice == '5':
            for symbol in symbols:
                try:
                    accessInstrument(symbol,instruments,5)
                except:
                    lg.error(f"INVALID-SYMBOL-{symbol}")
            time.sleep(2)
        elif choice == '6':
            for symbol in symbols:
                try:
                    accessInstrument(symbol,instruments,6)
                except:
                    lg.error(f"INVALID-SYMBOL-{symbol}")
            time.sleep(2)
        elif choice == '0':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")
            time.sleep(2)


if __name__ == "__main__":
    menu()
