import os
from flask import Flask, request, flash
from models import guess_cloth_category_model, guess_cloth_color_model, guess_cloth_brand_model

app = Flask(__name__)
#app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)),debug=True)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/')
def hello():
    return "Hello world"

@app.route('/cloth_category/train/', methods=['POST'])
def train_category():
    force_train = request.args.get('forceTrain', default = False, type = bool)
    return guess_cloth_category_model.train_category(force_train)

@app.route("/cloth_category/predict/", methods=['POST'])
def predict_category():
    if 'flask_file_field' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['flask_file_field']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        return guess_cloth_category_model.predict_category(file)
    return 'File is not allowed!'

@app.route('/cloth_color/train/', methods=['POST'])
def train_color():
    force_train = request.args.get('forceTrain', default = False, type = bool)
    return guess_cloth_color_model.train_color(force_train)

@app.route("/cloth_color/predict/", methods=['POST'])
def predict_color():
    if 'flask_file_field' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['flask_file_field']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        return guess_cloth_color_model.predict_color(file)
    return 'File is not allowed!'

@app.route('/cloth_brand/train/', methods=['POST'])
def train_brand():
    force_train = request.args.get('forceTrain', default = False, type = bool)
    return guess_cloth_brand_model.train_brand(force_train)

@app.route("/cloth_brand/predict/", methods=['POST'])
def predict_brand():
    if 'flask_file_field' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['flask_file_field']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        return guess_cloth_brand_model.predict_brand(file)
    return 'File is not allowed!'

def allowed_file(filename):     
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
