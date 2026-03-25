

README : How to run the assignments

This project contains implementations for two tasks. The first task involves learning word embeddings using Word2Vec on textual data collected from IIT Jodhpur sources. The second task focuses on character-level name generation using recurrent neural network models such as Vanilla RNN, Bidirectional LSTM and an attention-based RNN.

Before running the programs, the required Python libraries must be installed. This can be done using the command
pip install torch matplotlib numpy nltk gensim scikit-learn wordcloud

If any tokenizer related error occurs while running the code, the necessary NLTK resources can be downloaded by running the following commands once in Python.
import nltk
nltk.download('punkt')
nltk.download('stopwords')

To run the Word2Vec experiment(problem1), make sure that the textual dataset files are placed inside the data/raw_text folder. After ensuring the correct folder structure, execute the main script using the command
python main1.py

Running this script will automatically perform preprocessing of the collected text, train both CBOW and Skip-gram Word2Vec models, train the scratch implementation, and generate outputs such as nearest neighbour results, analogy observations, word cloud visualization and embedding plots. The cleaned corpus file and visualization images will also be saved in the project directory.

For the character-level name generation task (problem2), ensure that the dataset file TrainingNames.txt is present in the same folder as the program. The models can then be trained by running the command
python main.py


When executed, the program will train the Vanilla RNN, BLSTM and Attention-based RNN sequentially. It will display training loss values, generate sample names from each model and compute evaluation metrics such as novelty rate and diversity. A plot comparing the training behaviour of the models will also be displayed.


B23CM1047-A2 file has the corpus.txt and report.pdf which has all the outputs and plots

The evaluation of generated names is performed using simple Python functions that calculate how many names are new compared to the training dataset and how varied the generated samples are. These metrics help in comparing the performance of different architectures.



