import os
from flask import Flask, request
from models import guess_cloth_category_model

app = Flask(__name__)
#app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)),debug=True)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

@app.route('/')
def hello():
    return "Hello world"

@app.route('/cloth_type/train/', methods=['POST'])
def train():
    force_train = request.args.get('forceTrain', default = False, type = bool)
    return guess_cloth_category_model.train(force_train)

@app.route("/cloth_type/predict/", methods=['POST'])
def predict():
    if 'flask_file_field' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['flask_file_field']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        return guess_cloth_category_model.predict(file)
    return 'File is not allowed!'

def allowed_file(filename):     
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
