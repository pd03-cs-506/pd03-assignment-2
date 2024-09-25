# terminal run $ flask --app app --debug run
from flask import Flask, render_template, request, redirect, url_for
import csv
import os
from kmeans import *

app = Flask(__name__)

dataset = np.array([[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(300)])

# default display image
generate_image(dataset, 1, 'random', 0, 1)

@app.route("/",  methods=["GET", "POST"])
def home():
    # defaults:
    init_method = 'random'
    num_k = 2
    final = 0
    show_image = False 

    if request.method == "POST":
        init_method = request.form.get('init_method')
        num_k = request.form.get('num_k')

        if 'not-final' in request.form:
            final = 0
        elif 'final' in request.form:
            final = 1

        print(int(num_k))
        print(init_method)

        # def generate_image(dataset, k, init_method, reset_data, final):
        generate_image(dataset, int(num_k), init_method, 0, final)
        
        return redirect(url_for('home', show_image=True))

    show_image = request.args.get('show_image') == 'True'

    return render_template("index.html", 
                           init_method=init_method,
                           num_k=num_k,
                           show_image=show_image)

if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)