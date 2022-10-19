from pyngrok import ngrok 
public_url = ngrok.connect(port='8501')

# !streamlit run app.py & npx localtunnel --port 8501