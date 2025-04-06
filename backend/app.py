from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from Flask on Vercel!"

# Required for Vercel
if __name__ == "__main__":
    app.run()

# Vercel expects the variable 'app' to be present
app = app
