from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 Hello from Flask on Vercel!"

# Vercel requires the 'app' variable to be defined
app = app
