from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from utils import (
    load_model,
    load_tumor_model,
    predict_image,
    predict_tumor_type
)
from Chat import MedicalChatbot

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load primary model (aneurysm/cancer/tumor)
model, classes = load_model()

# Load secondary model (for tumor subtypes)
tumor_model, tumor_classes = load_tumor_model()
# Initialize chatbot instances for each session
chatbots = {}


def get_chatbot(session_id):
    """Get or create chatbot instance for session"""
    if session_id not in chatbots:
        api_key = os.getenv("GROQ_API_KEY", "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")
        chatbots[session_id] = MedicalChatbot(api_key)
    return chatbots[session_id]


@app.route('/')
def index():
    # Create unique session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No file selected'})

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in image.filename and
                image.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'})

        # Save uploaded image
        filename = secure_filename(image.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        image.save(filepath)

        # Predict using the model
        label, prob = predict_image(filepath, model, classes)

        tumor_subtype = None
        tumor_subtype_prob = None

        # If primary prediction is "tumor", use the secondary model
        if label.lower() == 'tumor':
            tumor_subtype, tumor_subtype_prob = predict_tumor_type(filepath, tumor_model, tumor_classes)

        # Get session and chatbot
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']

        chatbot = get_chatbot(session_id)

        # Set prediction context in chatbot
        chatbot.set_prediction_context(label, prob, datetime.now().isoformat())

        # Store prediction in session for reference
        session['last_prediction'] = {
            'label': label,
            'probability': prob,
            'timestamp': datetime.now().isoformat(),
            'filename': filename
        }

        response = {
            'label': label.capitalize(),
            'probability': f"{prob:.2%}",
            'confidence': f"{prob * 100:.2f}",
            'image_path': f"uploads/{filename}"
        }

        if tumor_subtype:
            response['tumor_subtype'] = tumor_subtype.capitalize()
            response['tumor_subtype_confidence'] = f"{tumor_subtype_prob:.2%}"

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'No message provided'})

        # Get session ID and chatbot instance
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']

        chatbot = get_chatbot(session_id)

        # Ensure prediction context is set if we have a recent prediction
        last_prediction = session.get('last_prediction')
        if last_prediction and not chatbot.active_prediction:
            chatbot.set_prediction_context(
                last_prediction['label'],
                last_prediction['probability'],
                last_prediction['timestamp']
            )

        # Generate response
        result = chatbot.generate_response(user_message)

        # Store conversation in chatbot history
        chatbot.conversation_history.append({
            "user_message": user_message,
            "bot_response": result['response'],
            "condition": result.get('condition'),
            "timestamp": result.get('timestamp'),
            "is_follow_up": result.get('is_follow_up', False),
            "aspects": result.get('aspects', []),
            "prediction_context": result.get('prediction_context')
        })

        return jsonify({
            'response': result['response'],
            'condition': result.get('condition'),
            'timestamp': result.get('timestamp'),
            'is_follow_up': result.get('is_follow_up', False),
            'has_prediction_context': chatbot.active_prediction is not None,
            'prediction_info': chatbot.active_prediction
        })

    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'})


@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for current session"""
    try:
        session_id = session.get('session_id')
        if not session_id or session_id not in chatbots:
            return jsonify({'history': []})

        chatbot = chatbots[session_id]
        history = chatbot.conversation_history[-20:]  # Last 20 exchanges

        return jsonify({
            'history': history,
            'current_topic': chatbot.current_topic,
            'active_prediction': chatbot.active_prediction,
            'total_conversations': len(chatbot.conversation_history)
        })

    except Exception as e:
        return jsonify({'error': f'Failed to get history: {str(e)}'})


@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history for current session"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in chatbots:
            chatbots[session_id].clear_history()  # This keeps prediction context

        return jsonify({'message': 'Chat history cleared successfully'})

    except Exception as e:
        return jsonify({'error': f'Failed to clear history: {str(e)}'})


@app.route('/chat/clear_all', methods=['POST'])
def clear_all_context():
    """Clear everything including prediction context"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in chatbots:
            chatbots[session_id].clear_all_context()

        # Also clear prediction from session
        if 'last_prediction' in session:
            del session['last_prediction']

        return jsonify({'message': 'All context cleared successfully'})

    except Exception as e:
        return jsonify({'error': f'Failed to clear all context: {str(e)}'})


@app.route('/session/info', methods=['GET'])
def get_session_info():
    """Get current session information"""
    try:
        session_id = session.get('session_id')
        last_prediction = session.get('last_prediction')
        chatbot_info = {}

        if session_id and session_id in chatbots:
            chatbot = chatbots[session_id]
            chatbot_info = {
                'current_topic': chatbot.current_topic,
                'active_prediction': chatbot.active_prediction,
                'conversation_count': len(chatbot.conversation_history),
                'topic_history': chatbot.topic_history[-5:]  # Last 5 topics
            }

        return jsonify({
            'session_id': session_id,
            'last_prediction': last_prediction,
            'chatbot_info': chatbot_info
        })

    except Exception as e:
        return jsonify({'error': f'Failed to get session info: {str(e)}'})


@app.route('/prediction/set', methods=['POST'])
def set_prediction_manually():
    """Manually set prediction context (for testing)"""
    try:
        data = request.get_json()
        label = data.get('label')
        confidence = data.get('confidence', 0.8)

        if not label or label.lower() not in ['cancer', 'aneurysm', 'tumor']:
            return jsonify({'error': 'Invalid label. Must be cancer, aneurysm, or tumor.'})

        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']

        chatbot = get_chatbot(session_id)
        chatbot.set_prediction_context(label, confidence)

        # Update session
        session['last_prediction'] = {
            'label': label,
            'probability': confidence,
            'timestamp': datetime.now().isoformat(),
            'filename': 'manual_prediction'
        }

        return jsonify({
            'message': f'Prediction context set to {label} with {confidence:.1%} confidence',
            'prediction_info': chatbot.active_prediction
        })

    except Exception as e:
        return jsonify({'error': f'Failed to set prediction: {str(e)}'})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error occurred.'}), 500


if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    print("🚀 Starting Integrated Medical Analysis System...")
    print("📊 Image prediction model loaded")
    print("🤖 Enhanced chatbot system ready")
    print("🔍 Prediction context tracking enabled")
    print("🌐 Web interface available at http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)