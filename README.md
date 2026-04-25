This project is a deep learning–powered web application that classifies fish species from an uploaded image.
Users can upload a fish image through a browser interface, and the system returns the predicted species along with confidence levels.
The model is trained using transfer learning with MobileNetV2 and deployed using Flask.

Steps: 
1. Download all files (or the zip) and ensure it's in this structure in your IDE
   
*Project Folder*/

│

├── data/

│   ├── train/

│   └── val/

│

├── models/

│   └── fish_classifier.keras (Not downloadable)

│

├── templates/

│   └── index.html

│

├── app.py

├── train.py

├── requirements.txt

└── README.md

3. Install dependencies if necessary in Terminal: pip install -r requirements.txt
4. Run train.py to generate the fish_classifier.keras file (this may take several minutes)
5. Start the Flask server: python app.py
6. Paste either the address given by the program, or this line, into your browser: http://127.0.0.1:5000
7. There will be a UI prompting the upload of an image. Upload your image.
8. The program will return a prediction and its confidence estimate.
