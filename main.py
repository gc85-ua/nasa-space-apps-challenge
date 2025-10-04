from flask import Flask, render_template

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static")

@app.route('/')
def inicio():
    return render_template('Inicio.html')

@app.route('/buscar')
def buscar():
    return render_template('buscar.html')

@app.route('/detalles')
def detalles():
    return render_template('detalles.html')


@app.route('/mapa')
def mapa():
    return render_template('mapa.html')

@app.route('/comunidad')
def comunidad():
    return render_template('comunidad.html')

if __name__ == '__main__':
    app.run(debug=True)
