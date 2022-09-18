from flask import (
	Flask,
	make_response
)

app = Flask(__name__)
app.config["DEBUG"] = True


from sklearn.feature_extraction.text import CountVectorizer


@app.route("/test", methods=["GET"])
def on_test():
	return make_response("Test route hit!", 200)

@app.route('/generate', methods=['POST'])
def generate_ticket():
	try:
		return make_response("GENERATE TICKET ROUTE!", 200)
	except Exception as e:
		make_response(repr(e), 500)

if __name__ == "__main__":
	app.run()
	