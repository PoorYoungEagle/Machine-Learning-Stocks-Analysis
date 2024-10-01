import streamlit as st
from CraneML import ticker
from CraneML import plot_close_prices
from CraneML import plot_open_prices
from CraneML import MLModel


#The main file which uses streamlit to provide a user interface for the models.
#Run using "streamlit run Cranelit.py" in the terminal.


def main():
      
      #Provides options for the type of stocks
      #Can be changed by adding in more tickers into the 'options' list
      st.title('Crane')
      st.subheader("Ticker")
      options = ['NVDA','AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
      selected_stock = st.selectbox('Choose a stock:', options)

      #Returns the values required to add in the visual representation of Open Prices, Close Prices and Accuracy Graph
      df, open_prices, close_price, close_prices, high_prices, low_prices, volumes = ticker(selected_stock)

      #Open Prices
      st.subheader("Open Prices Plot")
      Plot_Open = plot_open_prices(open_prices)
      st.pyplot(Plot_Open)
      
      #Close Prices
      st.subheader("Close Prices Plot")
      Plot_Close = plot_close_prices(close_prices)
      st.pyplot(Plot_Close)

      #Streamlit Slider with a maximum of 2000 epochs
      st.subheader("Machine Learning")
      epochs = st.slider('Select number of epochs:', min_value=1, max_value=2000, value=100)

      #Progress bar for completion
      progress_bar = st.empty()
      def Progress_Update(progress):
            progress_bar.progress(progress)
      loss=0.0
      accuracy=0.0
      
      #Button to activate the machine learning code
      if st.button('Start Training'):
            st.session_state.training = True
            AccuracyG,loss,accuracy,history=MLModel(Progress_Update, epochs)
            st.success("Training completed!")
            st.write(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}")
            st.pyplot(AccuracyG)

if __name__ == "__main__":
      main()