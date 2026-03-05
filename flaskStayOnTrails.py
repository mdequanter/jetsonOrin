from flask import Flask, render_template, redirect, url_for

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.get("/")
def home():
    # zelfde als jouw "Home" link; stuur door naar try-it of maak eigen home template
    return redirect(url_for("stayontrails"))


@app.get("/stayontrails")
def stayontrails():
    # Render de pagina (HTML/JS blijft client-side)
    return render_template("stayontrails.html")


if __name__ == "__main__":
    # If you use SSL like before, keep it. Otherwise remove ssl_context for local testing.
    app.run(
        host="0.0.0.0",
        port=5001,
        ssl_context=("jetson-desktop+5.pem", "jetson-desktop+5-key.pem"),
        debug=False
    )