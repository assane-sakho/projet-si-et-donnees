from flask import Flask, request
from models import guess_cloth_category_model

app = Flask(__name__)
# app.run(host='0.0.0.0', port=8080,debug=True)

@app.route('/')
def hello():
    return "Hello world"

@app.route('/cloth_type/train/')
def train():
    return guess_cloth_category_model.train()

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
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return f'Uploaded {file.filename}'
    return 'File is not allowed!'

def test():  
    if request.method == 'POST':
       
        if 'file' not in request.files:
            print('No file part')
            return 'No file part', 500
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return 'No selected file', 50
        if file and allowed_file(file.filename):
            print('predict')
            return guess_cloth_category_model.predict(file)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
