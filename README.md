Problem statement :
Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness among diabetic patients worldwide. 
Early detection and accurate severity grading are essential for timely treatment and vision preservation. However, the current diagnosis process relies on manual examination of retinal fundus images by ophthalmologists, which is time-consuming and subjective.
In addition, different ophthalmologists may provide different severity assessments for the same retinal image, leading to variations in diagnosis. Furthermore, existing automated approaches mainly focus on DR detection rather than detailed severity grading and often lack clinical interpretability. 
Therefore, there is a need for an automated, accurate, and explainable deep learning-based system that can classify DR severity stages from retinal fundus images and assist clinicians in making reliable diagnostic decisions.


Proposed solution:
To address the limitations of manual Diabetic Retinopathy (DR) diagnosis, an automated and explainable deep learning-based system is proposed for DR severity grading using retinal fundus images. 
The proposed framework utilizes a CNN architecture such as DenseNet121 or ResNet to extract important retinal features associated with DR. 
An attention mechanism is incorporated to focus on clinically significant lesion regions, thereby improving classification performance.
The system classifies retinal images into five severity stages: No DR, Mild, Moderate, Severe, and Proliferative DR. To enhance clinical trust and interpretability, Grad-CAM is employed to highlight the retinal regions that contribute to the model's prediction. 
Additionally, the framework provides referral recommendations based on the predicted severity level, supporting timely clinical intervention.
