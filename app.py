import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
#from scipy import misc
from skimage import io
from scipy.misc import imread, imresize,imshow
from keras import backend as K

from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
	if request.method=='POST':
        
        
           file=request.files['image']
           if not file: return render_template('index.html',label="No file")
           #img=misc.imread(file)
           # dimensions of our images
           img_width, img_height = 224,224
           filename="model_saved_improved_full.h5"
           #loaded_model = pickle.load(open(filename, 'rb'))
           model.load_weights(filename)
           model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

           # predicting images
           img = image.load_img(file, target_size=(img_width, img_height))
           x = image.img_to_array(img)
           x = np.expand_dims(x, axis=0)

           images = np.vstack([x])
           w_class = model.predict_classes(images, batch_size=1)
           print(w_class)
           label=str(np.squeeze(w_class))
           
	return render_template('index.html', label=label)
if __name__ == '__main__':
	# load ml model
	#model = joblib.load('model.pkl')
	# start api
	app.run(host='127.0.0.1', port=1801, debug=False,use_reloader=False)
