import os
import shutil
import zipfile


# Unzip.
zip_file_path = 'Pretrained_Source.zip'
extract_path = './'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

if os.path.exists('./__MACOSX'):
    shutil.rmtree('./__MACOSX')

# Process.
os.makedirs('./Data/Trinity', exist_ok=True)
if os.path.exists('./Data/Trinity/Processed'):
    shutil.rmtree('./Data/Trinity/Processed')
shutil.move('./Pretrained_Source/Processed', './Data/Trinity/')

os.makedirs('./Gesture_Lexicon/Training/Trinity/Conv1d/Checkpoints', exist_ok=True)
os.makedirs('./Gesture_Lexicon/Training/Trinity/Transformer/Checkpoints', exist_ok=True)
os.rename('./Pretrained_Source/Gesture_Lexicon/trained_model_Conv1d.pth', 
          './Gesture_Lexicon/Training/Trinity/Conv1d/Checkpoints/trained_model.pth')
os.rename('./Pretrained_Source/Gesture_Lexicon/trained_model_Transformer.pth', 
          './Gesture_Lexicon/Training/Trinity/Transformer/Checkpoints/trained_model.pth')

os.makedirs('./Gesture_Generator/Training/Trinity/RNN/Checkpoints', exist_ok=True)
os.rename('./Pretrained_Source/Gesture_Generator/trained_model.pth', 
          './Gesture_Generator/Training/Trinity/RNN/Checkpoints/trained_model.pth')

os.makedirs('./Lexeme_Interpreter/Training/Trinity/Transformer/Checkpoints', exist_ok=True)
os.rename('./Pretrained_Source/Lexeme_Interpreter/trained_model.pth', './Lexeme_Interpreter/Training/Trinity/Transformer/Checkpoints/trained_model.pth')

shutil.rmtree('./Pretrained_Source')