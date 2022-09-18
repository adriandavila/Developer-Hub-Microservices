from flask import (
	Flask,
	request,
	make_response
)

doc = '''"Okay, so when linking one element to another, we'll need to use an overlay of some sort to grab and query elements related to the one being written. This overlay has some other major components. This GUI element needs to pop up when a user types something, clicks something, hits a hotkey, a button, etc. We'll link with @Hugo Monday for a quick talk during the 1300 meeting and find out what he wants the user experience to be. The linkage system is done, so we just need the creation piece. Some general things to note: This has to be visible in the  TemplateModal  menu It has to insert the chosen  element   data   uuid  within a pair of double curly braces: {{<linked data uuid>}} When relating an element, we need to do two things Add the  linked  key to element metadata with a link to any other elements linked via the  element.data  keys. It will init as an empty array. When submitting an element, if a user  links  element data to another element's data, we need to add the new Elements UUID to the one being linked. That sounds insanely complicated but I'll explain it when you ask. This ticket will require smaller sequential tickets, don't be afraid to seek out more help direction. It's a little blank now but we'll tackle it on Monday with Hugo."'''

app = Flask(__name__)
app.config["DEBUG"] = True


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import itertools

model = SentenceTransformer('distilbert-base-nli-mean-tokens')

h2a_stop_words = frozenset(["ticket", "tickets", "new","monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "week", "1300", "hugo", "karl", "clint", "meeting", "quick"])


@app.route("/test", methods=["GET"])
def on_test():
	return make_response("Test route hit!", 200)

@app.route('/generate', methods=['GET'])
def generate_ticket():
	try:
		request_data = request.get_json()

		n_gram_range = (1, 10)
		stop_words = "english"

		# Extract candidate words/phrases
		count = CountVectorizer(ngram_range=n_gram_range, stop_words=h2a_stop_words).fit([doc])
		candidates = count.get_feature_names_out()

		doc_embedding = model.encode([doc])
		candidate_embeddings = model.encode(candidates)

		top_n = 5
		distances = cosine_similarity(doc_embedding, candidate_embeddings)
		keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

		return make_response({"title_ideas": keywords}, 200)
	except Exception as e:
		make_response(repr(e), 500)

if __name__ == "__main__":
	app.run()
	