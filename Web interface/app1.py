from flask import Flask, render_template, request
from model import predict_data

app=Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method=="POST":
        form_data = request.form.to_dict()
        form_data['Over18'] = 'Y'
        for key in form_data:
            form_data[key] = [form_data[key]]
        predictions, accuracy = predict_data(form_data)
        return render_template('result.html', attrition=predictions[0], accuracy=accuracy)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=3000)