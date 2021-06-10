from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import os
import shutil
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy import spatial
# import torchvision
import time
from Logger.Logger import logger
from pathlib import Path
ALLOWED_EXTENSIONS = {'jpg','jpeg','png'}

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

testImagePath= "testImages"

mtcnn = MTCNN(160, margin=0)
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
#
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS





@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def index():
    imagesEmbed = []
    try:
        if (request.files['baselineImage'].filename != "") and (request.files['submittedImage'].filename != "") and allowed_file(
                request.files['baselineImage'].filename):

            start_time = time.time()
            baselineImage = request.files['baselineImage']
            submittedImage=request.files['submittedImage']
            baselineImagePath = os.path.join(os.getcwd(), 'testImages/', '1'+baselineImage.filename)
            submittedImagePath = os.path.join(os.getcwd(), 'testImages/', '2'+submittedImage.filename)
            baselineImage.save(baselineImagePath)
            submittedImage.save(submittedImagePath)
            logger.info("Images saved Successfully...!!!")

            for images in os.listdir('testImages/'):
                img = Image.open('testImages/' + "/" + images)

                # Get cropped and prewhitened image tensor
                img_cropped = mtcnn(img)
                # img_cropped.save("cropped.jpg")
                # print(img_cropped)

                if img_cropped== None:
                    shutil.rmtree(testImagePath)
                    Path(testImagePath).mkdir(parents=True, exist_ok=True)
                    return jsonify({"result": "In-appropriate File passed! No face found"}), 200
                else:
                    #for saving the cropped image
                    # torchvision.utils.save_image(img_cropped, "./croppedImage/"+images)
                    pass


                    # Calculate embedding (unsqueeze to add batch dimension)
                    img_embedding = resnet(img_cropped.unsqueeze(0))
                    imagesEmbed.append((img_embedding.tolist())[0])

            shutil.rmtree(testImagePath)
            Path(testImagePath).mkdir(parents=True, exist_ok=True)
            cosine_similarity = (1 - spatial.distance.cosine(imagesEmbed[0], imagesEmbed[1])) * 100
            logger.info("Similarity Score: "+str(cosine_similarity))
            similarityScore= 0 if cosine_similarity<0 else cosine_similarity
            result= 'Matched' if similarityScore>= 50.00 else 'Not Matched'
            respTime= (time.time() - start_time)
            return jsonify({"result": result, "score": similarityScore, "time":respTime}), 200
        else:
            return jsonify({"message":"In-appropriate File passed please pass only '{'jpg','jpeg','png'}' file...!", "status": "fail"}), 415
    except Exception as ex:
        print(ex)
        return jsonify(
            {"message": "In-appropriate File passed please pass only '{'jpg','jpeg','png'}' file...!", "status": "fail"}), 415



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
#    Path(testImagePath).mkdir(parents=True, exist_ok=True)
    os.mkdir(testImagePath) if not os.path.isdir(testImagePath) else None
