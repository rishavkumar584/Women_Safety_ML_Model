# Women_Safety_ML_Model
This model is a Real-Time webcam-based safety system which can detect the Help Hand Symbol of a woman,
and logs the event with a timestamp. Made as an AI-powered safety alert system, which can be integrated into surveillance, security, or emergency response systems.

The model was trained upon a total of 2307 human face images( 1173 male + 1134 female ) for better recognization & differentiation of men and women.

## Porject Features
- **Live Webcam Feed Monitoring** using OpenCV.
- Counts the no. of people ( no. of men and no. of woman in the frame ).
- Detects alone woman surrounded by men.
- Detects the **HELP!** hand gesture.
- Generates log incase help symbol is initiated by woman.
- The generated log is read by simple webpage interface for better representation.

  ## Future updates
  Configuring with real time location capture.
  Sending of alert msgs on family & nearest police station by detecting womans information from face detection.
  Improving further to detect face expression & voice captures methods to avoid the need of requiring the help gesture notation.
   
