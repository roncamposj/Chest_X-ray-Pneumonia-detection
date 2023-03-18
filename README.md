# Chest_X-ray-Pneumonia-detection

Perform the following steps to run this code:
1. clone repository
2. pip install -r requirements.txt

To train model, run train_model.py with desired flags. To train model from scratch, set --mode scratch.  For a transfer learning, set --mode pretrained.  An example:
1. python -m train_model --mode scratch --batch_size 32 --num_epochs 100 --dropout 0.3 

scratch-model.pth and pretrained-model.pth are the saved scratch and pretrained models, respectively.  You can view their performances in accuracies.ipynb
