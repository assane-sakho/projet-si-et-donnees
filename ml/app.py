from flask import Flask, request
app = Flask(__name__)
# app.run(host='0.0.0.0', port=8080,debug=True)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route("/guessClotheType/", methods=['POST'])
def guess_cloth_type():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # # if user does not select file, browser also
        # # submit a empty part without filename
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)       
        # if file and allowed_file(file.filename):
            # return "t-shirt"
        return "t-shirt"