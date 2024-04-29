# TravelPal - Your Personal Travel Assistant 🌍🧳

TravelPal is a chatbot designed to assist users in planning their travel itinerary. It provides recommendations for accommodations, activities, and restaurants based on user preferences. The chatbot uses Natural Language Understanding (NLU) techniques to understand user input and respond appropriately.

## Features
- **Intent Recognition:** TravelPal identifies the user's intent (e.g., find accommodation, recommend activities) using a combination of TF-IDF vectorization and Support Vector Machine (SVM) classification.
- **Entity Extraction:** It extracts relevant entities (e.g., lodging type, activity type, cuisine type) from user input using spaCy.
- **Context Handling:** TravelPal maintains context across conversations to provide more personalized recommendations.
- **Dialog Management:** It manages the conversation flow, guiding users through the planning process effectively.
- **Iterative Improvement:** The system continuously learns and improves based on evaluation results, adjusting models and dialog management approaches to enhance performance.
- **Model Evaluation:** TravelPal evaluates intent recognition and entity extraction models separately using metrics such as accuracy, precision, recall, and F1 score. It also tests the complete NLU module in a simulated dialog scenario to assess its effectiveness.

## Technologies Used
- Python
- scikit-learn
- spaCy
- Streamlit

## How to Use
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run app.py`.
4. Start chatting with TravelPal by typing your queries and preferences.

## Iterative Improvement
TravelPal's performance can be enhanced through iterative improvements:
- Based on evaluation results, adjust the models or the dialog management approach to improve the NLU module's performance.
- Experiment with different techniques (e.g., BERT-based models for intent recognition) to enhance the system's capabilities.

## Model Evaluation
Evaluate the intent recognition and entity extraction models separately using metrics such as accuracy, precision, recall, and F1 score. Test the complete NLU module in a simulated dialog scenario to assess its effectiveness in guiding the dialog system.

Feel free to contribute to the project by suggesting enhancements, reporting issues, or submitting pull requests!
