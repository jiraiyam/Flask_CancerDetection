import requests
import json
import os
import re
from datetime import datetime


class EnhancedMedicalChatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Enhanced medical information database
        self.medical_knowledge_base = {
            "cancer": {
                "summary": "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body.",
                "types": ["Breast Cancer", "Lung Cancer", "Prostate Cancer", "Colorectal Cancer", "Skin Cancer",
                          "Blood Cancers"],
                "common_symptoms": ["Unexplained weight loss", "Persistent fatigue", "Unusual lumps or swelling",
                                    "Changes in skin appearance", "Persistent cough or hoarseness"],
                "risk_factors": ["Age", "Family history", "Smoking", "Excessive alcohol", "Exposure to radiation",
                                 "Certain chemicals"],
                "prevention": ["Regular screenings", "Healthy diet", "Regular exercise", "Avoid smoking",
                               "Limit alcohol", "Sun protection"],
                "urgency_level": "High - Requires immediate medical attention for proper diagnosis and treatment"
            },
            "aneurysm": {
                "summary": "An aneurysm is a bulge or ballooning in a blood vessel caused by weakness in the vessel wall.",
                "types": ["Aortic Aneurysm", "Brain Aneurysm", "Peripheral Aneurysm", "Thoracic Aneurysm",
                          "Abdominal Aneurysm"],
                "common_symptoms": ["Sudden severe headache", "Chest or back pain", "Pulsating feeling in abdomen",
                                    "Vision problems", "Nausea and vomiting"],
                "risk_factors": ["High blood pressure", "Smoking", "Family history", "Age over 65", "Atherosclerosis",
                                 "Connective tissue disorders"],
                "prevention": ["Control blood pressure", "Don't smoke", "Regular exercise", "Healthy diet",
                               "Manage cholesterol", "Regular check-ups"],
                "urgency_level": "Critical - Can be life-threatening, requires immediate medical evaluation"
            },
            "tumor": {
                "summary": "A tumor is an abnormal growth of tissue that can be benign (non-cancerous) or malignant (cancerous).",
                "types": ["Benign Tumors", "Malignant Tumors", "Brain Tumors", "Soft Tissue Tumors", "Bone Tumors"],
                "common_symptoms": ["Unexplained lumps", "Pain or pressure", "Neurological symptoms",
                                    "Changes in function", "Swelling or inflammation"],
                "risk_factors": ["Age", "Genetics", "Environmental factors", "Previous radiation",
                                 "Immune system disorders", "Hormonal factors"],
                "prevention": ["Regular health screenings", "Healthy lifestyle", "Avoid known carcinogens",
                               "Maintain healthy weight", "Exercise regularly"],
                "urgency_level": "Moderate to High - Requires medical evaluation to determine if benign or malignant"
            }
        }

        # Enhanced conditions with related terms
        self.medical_conditions = {
            "cancer": ["cancer", "carcinoma", "malignancy", "oncology", "chemotherapy",
                       "radiation therapy", "metastasis", "tumor malignant", "cancerous"],
            "aneurysm": ["aneurysm", "aneurism", "arterial dilation", "vascular",
                         "aortic aneurysm", "brain aneurysm", "cerebral aneurysm"],
            "tumor": ["tumor", "tumour", "neoplasm", "growth", "mass", "benign tumor",
                      "malignant tumor", "cyst", "lesion"]
        }

        # Context-aware keywords for follow-up questions
        self.context_keywords = {
            "summary": ["summary", "overview", "what is", "explain", "definition", "about"],
            "types": ["types", "type", "kinds", "kind", "categories", "varieties", "forms"],
            "symptoms": ["symptoms", "symptom", "signs", "indication", "manifestation", "warning signs"],
            "treatment": ["treatment", "therapy", "cure", "medication", "surgery", "procedure", "options"],
            "causes": ["causes", "cause", "reason", "why", "how", "origin", "risk factors"],
            "diagnosis": ["diagnosis", "diagnose", "test", "detection", "screening", "how to detect"],
            "prevention": ["prevention", "prevent", "avoid", "reduce risk", "protect", "lifestyle"],
            "prognosis": ["prognosis", "outcome", "survival", "recovery", "life expectancy", "outlook"],
            "advice": ["advice", "recommend", "should do", "next steps", "action", "help"],
            "urgent": ["urgent", "emergency", "serious", "dangerous", "worry", "scared"]
        }

        self.system_prompt = (
            "You are an advanced medical information assistant with integrated image analysis capabilities. "
            "You specialize in THREE medical conditions: CANCER, ANEURYSM, and TUMOR. "
            "CRITICAL INSTRUCTIONS: "
            "1. ALWAYS reference the AI image analysis prediction when available "
            "2. Provide structured, comprehensive information including: summary, symptoms, causes, treatment options, and actionable advice "
            "3. Use clear formatting with headers and bullet points for better readability "
            "4. Include urgency level and next steps recommendations "
            "5. Be compassionate but informative, acknowledging the user's concerns "
            "6. Always include appropriate medical disclaimers "
            "7. If asked about other conditions, politely redirect to the three allowed topics "
            "8. Structure responses as: Summary → Key Information → Practical Advice → Next Steps "
            "REMEMBER: All information is educational only - users must consult healthcare professionals."
        )

        # Store conversation history and context
        self.conversation_history = []
        self.current_topic = None
        self.topic_history = []
        self.prediction_context = None
        self.active_prediction = None

    def set_prediction_context(self, label, confidence, timestamp=None):
        """Set context from image prediction results"""
        self.prediction_context = {
            "label": label.lower(),
            "confidence": confidence,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        if label.lower() in self.medical_conditions:
            self.current_topic = label.lower()
            self.active_prediction = self.prediction_context.copy()

    def get_structured_medical_info(self, condition, aspects=None):
        """Get structured medical information for a condition"""
        if condition not in self.medical_knowledge_base:
            return None

        info = self.medical_knowledge_base[condition]

        # If specific aspects are requested, focus on those
        if aspects:
            structured_info = {}
            for aspect in aspects:
                if aspect == "summary" and "summary" in info:
                    structured_info["summary"] = info["summary"]
                elif aspect == "types" and "types" in info:
                    structured_info["types"] = info["types"]
                elif aspect == "symptoms" and "common_symptoms" in info:
                    structured_info["symptoms"] = info["common_symptoms"]
                elif aspect == "causes" and "risk_factors" in info:
                    structured_info["risk_factors"] = info["risk_factors"]
                elif aspect == "prevention" and "prevention" in info:
                    structured_info["prevention"] = info["prevention"]
            return structured_info

        return info

    def generate_prediction_summary(self, condition, confidence):
        """Generate a comprehensive summary based on prediction"""
        info = self.get_structured_medical_info(condition)
        if not info:
            return "Unable to generate summary for this condition."

        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"

        summary = f"""
🔍 **ANALYSIS SUMMARY**
─────────────────────────────────────
**Condition Detected:** {condition.upper()}
**Confidence Level:** {confidence:.1%} ({confidence_text} confidence)
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

📋 **CONDITION OVERVIEW**
{info['summary']}

⚠️ **URGENCY LEVEL**
{info['urgency_level']}

🎯 **KEY SYMPTOMS TO MONITOR**
• {' • '.join(info['common_symptoms'][:3])}
• And other related symptoms

🛡️ **IMMEDIATE RECOMMENDATIONS**
1. **Consult a Healthcare Professional** - Schedule an appointment with your doctor immediately
2. **Document Symptoms** - Keep track of any symptoms you're experiencing
3. **Gather Medical History** - Prepare your family medical history and current medications
4. **Stay Calm** - Early detection often leads to better outcomes

⚕️ **NEXT STEPS**
• Book an appointment with your primary care physician
• Consider getting a second opinion if needed
• Follow up with appropriate specialists as recommended
• Maintain a healthy lifestyle while awaiting professional evaluation

**DISCLAIMER:** This AI analysis is for informational purposes only and should not replace professional medical diagnosis. Please consult with qualified healthcare providers for proper evaluation and treatment.
"""
        return summary

    def generate_detailed_advice(self, condition, user_question):
        """Generate detailed, actionable advice"""
        info = self.get_structured_medical_info(condition)
        if not info:
            return "Unable to provide detailed advice for this condition."

        advice = f"""
💡 **DETAILED INFORMATION: {condition.upper()}**
─────────────────────────────────────

📖 **COMPREHENSIVE OVERVIEW**
{info['summary']}

🔍 **COMMON TYPES**
• {' • '.join(info['types'][:4])}

🚨 **WARNING SYMPTOMS**
• {' • '.join(info['common_symptoms'])}

⚠️ **RISK FACTORS**
• {' • '.join(info['risk_factors'])}

🛡️ **PREVENTION STRATEGIES**
• {' • '.join(info['prevention'])}

📅 **RECOMMENDED ACTIONS**
1. **Immediate:** Schedule medical consultation within 24-48 hours
2. **Short-term:** Complete recommended diagnostic tests
3. **Ongoing:** Follow healthcare provider's treatment plan
4. **Long-term:** Maintain regular follow-up appointments

🔴 **WHEN TO SEEK EMERGENCY CARE**
• Sudden severe symptoms
• Rapid worsening of condition
• Signs of complications
• Any life-threatening symptoms

💪 **SUPPORT RESOURCES**
• Connect with support groups
• Consider counseling services
• Involve family/friends in care planning
• Research reputable medical resources
"""
        return advice

    def clean_response(self, response_text):
        """Remove <think> sections and clean up the response"""
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
        return cleaned_response.strip()

    def detect_context_keywords(self, user_input):
        """Detect what aspect of a topic the user is asking about"""
        user_input_lower = user_input.lower()
        detected_aspects = []

        for aspect, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    detected_aspects.append(aspect)
                    break

        return detected_aspects

    def is_follow_up_question(self, user_input):
        """Check if this is a follow-up question about the current topic"""
        if not self.current_topic:
            return False

        user_input_lower = user_input.lower()
        follow_up_indicators = [
            "what are the", "what is the", "tell me about", "how about",
            "what about", "can you explain", "more about", "details about",
            "types", "symptoms", "treatment", "causes", "how serious",
            "should i worry", "what should i do", "is this dangerous",
            "explain", "tell me", "what", "how", "why", "when", "advice"
        ]

        for indicator in follow_up_indicators:
            if indicator in user_input_lower:
                return True

        return len(user_input_lower.split()) <= 6

    def is_valid_medical_topic(self, user_input):
        """Check if the user's question is about our supported medical conditions"""
        user_input_lower = user_input.lower()

        for condition, keywords in self.medical_conditions.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    return condition

        if self.is_follow_up_question(user_input):
            return self.current_topic

        if self.active_prediction and not any(condition in user_input_lower
                                              for condition in self.medical_conditions.keys()):
            return self.active_prediction["label"]

        return None

    def build_enhanced_prompt(self, user_input, detected_condition):
        """Build an enhanced prompt with structured information requests"""
        aspects = self.detect_context_keywords(user_input)

        # Check if user wants summary or general information
        if any(aspect in aspects for aspect in ["summary", "advice", "urgent"]):
            if self.active_prediction:
                return f"""
PREDICTION CONTEXT: User received AI analysis showing '{self.active_prediction['label'].upper()}' 
with {self.active_prediction['confidence']:.1%} confidence.

User is asking for: {', '.join(aspects) if aspects else 'general information'}

Provide a comprehensive, well-structured response about {detected_condition.upper()} that includes:
1. Clear summary referencing the AI prediction
2. Key symptoms and warning signs
3. Practical advice and next steps
4. Urgency level and when to seek care
5. Supportive, compassionate tone

User question: {user_input}
"""

        # Build contextual prompt for specific questions
        base_prompt = user_input
        if self.active_prediction:
            prediction_context = (
                f"PREDICTION CONTEXT: User's image analysis detected '{self.active_prediction['label'].upper()}' "
                f"with {self.active_prediction['confidence']:.1%} confidence. "
            )
            base_prompt = prediction_context + "User question: " + base_prompt

        return base_prompt

    def generate_response(self, user_input):
        """Generate enhanced response with structured medical information"""
        # Detect if question is about supported conditions
        detected_condition = self.is_valid_medical_topic(user_input)

        if not detected_condition:
            if (self.active_prediction and
                    any(word in user_input.lower() for word in
                        ["result", "analysis", "prediction", "found", "detected", "image", "scan", "summary"])):
                detected_condition = self.active_prediction["label"]
            else:
                return self._generate_redirect_response()

        # Update topic context
        if detected_condition and detected_condition != self.current_topic:
            self.current_topic = detected_condition

        # Check for summary/advice requests
        aspects = self.detect_context_keywords(user_input)

        # Generate structured response for summary requests
        if ("summary" in aspects or "advice" in aspects or
                any(word in user_input.lower() for word in
                    ["summary", "overview", "tell me about", "what should i do"])):

            if self.active_prediction:
                structured_response = self.generate_prediction_summary(
                    detected_condition, self.active_prediction['confidence']
                )
                return {
                    "response": structured_response,
                    "condition": detected_condition,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "is_summary": True,
                    "prediction_context": self.active_prediction
                }

        # Generate detailed advice for specific questions
        if ("advice" in aspects or "urgent" in aspects or
                any(word in user_input.lower() for word in
                    ["what should i do", "help", "scared", "worried", "dangerous"])):
            detailed_response = self.generate_detailed_advice(detected_condition, user_input)
            return {
                "response": detailed_response,
                "condition": detected_condition,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_detailed_advice": True,
                "prediction_context": self.active_prediction
            }

        # Generate AI-powered response for other questions
        return self._generate_ai_response(user_input, detected_condition, aspects)

    def _generate_redirect_response(self):
        """Generate response for non-supported topics"""
        current_topic_text = f"Currently discussing: {self.current_topic.upper()}. " if self.current_topic else ""
        prediction_text = f"Your recent image analysis detected: {self.active_prediction['label'].upper()}. " if self.active_prediction else ""

        return {
            "response": (
                "🏥 **MEDICAL ASSISTANT SCOPE** 🏥\n\n"
                "I specialize in providing information about three specific medical conditions:\n\n"
                "🔹 **CANCER** - Types, symptoms, treatments, prevention\n"
                "🔹 **ANEURYSM** - Causes, symptoms, treatments, risk factors\n"
                "🔹 **TUMOR** - Benign vs malignant, symptoms, treatments\n\n"
                f"{prediction_text}"
                f"{current_topic_text}"
                "**💡 Try asking:**\n"
                "• 'Give me a summary of my results'\n"
                "• 'What should I do about this?'\n"
                "• 'What are the symptoms?'\n"
                "• 'How serious is this condition?'\n\n"
                "I'm here to provide comprehensive, structured information about these conditions!"
            ),
            "condition": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_redirect": True
        }

    def _generate_ai_response(self, user_input, detected_condition, aspects):
        """Generate AI-powered response using the API"""
        enhanced_prompt = self.build_enhanced_prompt(user_input, detected_condition)

        messages = [{"role": "system", "content": self.system_prompt}]

        if self.active_prediction:
            prediction_system_msg = (
                f"CRITICAL: User has AI image analysis result showing '{self.active_prediction['label'].upper()}' "
                f"with {self.active_prediction['confidence']:.1%} confidence. "
                f"ALWAYS reference this prediction and provide structured, actionable information."
            )
            messages.append({"role": "system", "content": prediction_system_msg})

        # Add conversation context
        for exchange in self.conversation_history[-3:]:
            if exchange.get("user_message") and exchange.get("bot_response"):
                messages.append({"role": "user", "content": exchange["user_message"]})
                messages.append({"role": "assistant", "content": exchange["bot_response"]})

        messages.append({"role": "user", "content": enhanced_prompt})

        data = {
            "model": "deepseek-r1-distill-llama-70b",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 800,
            "top_p": 0.9
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()

            bot_response = response.json()["choices"][0]["message"]["content"]
            bot_response = self.clean_response(bot_response)

            # Add prediction reference if not included
            if (self.active_prediction and
                    detected_condition == self.active_prediction["label"] and
                    self.active_prediction["label"].lower() not in bot_response.lower()):
                prediction_reference = (
                    f"\n\n📊 **ANALYSIS CONTEXT**\n"
                    f"This information relates to your image analysis showing "
                    f"'{self.active_prediction['label'].upper()}' with "
                    f"{self.active_prediction['confidence']:.1%} confidence."
                )
                bot_response += prediction_reference

            return {
                "response": bot_response,
                "condition": detected_condition,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success",
                "is_follow_up": self.is_follow_up_question(user_input),
                "aspects": aspects,
                "prediction_context": self.active_prediction
            }

        except requests.exceptions.Timeout:
            return {
                "response": "⏰ Request timed out. Please try again.",
                "condition": detected_condition,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error"
            }
        except requests.exceptions.HTTPError as err:
            return {
                "response": f"🚨 API Error ({err.response.status_code}). Please try again later.",
                "condition": detected_condition,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error"
            }
        except Exception as e:
            return {
                "response": f"❌ Unexpected error: {str(e)}. Please try again.",
                "condition": detected_condition,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "error"
            }

    def clear_history(self):
        """Clear conversation history but keep prediction context"""
        self.conversation_history.clear()
        self.topic_history.clear()

    def clear_all_context(self):
        """Clear everything including prediction context"""
        self.conversation_history.clear()
        self.current_topic = None
        self.topic_history.clear()
        self.prediction_context = None
        self.active_prediction = None


# Keep the MedicalChatbot class name for compatibility
class MedicalChatbot(EnhancedMedicalChatbot):
    pass


def main():
    """Main function for standalone chatbot usage"""
    print("🚀 Starting Enhanced Medical Information Chatbot...")
    api_key = os.getenv("GROQ_API_KEY", "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")

    if not api_key or api_key == "your_api_key_here":
        print("❌ Error: Please set your GROQ API key!")
        return

    try:
        chatbot = EnhancedMedicalChatbot(api_key)
        print("\n🏥 Enhanced Medical Assistant Ready!")
        print("💡 Try: 'Give me a summary' or 'What should I do?' after image analysis")

        while True:
            user_input = input("\n🔹 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for using the Enhanced Medical Assistant!")
                break

            result = chatbot.generate_response(user_input)
            print(f"\n🤖 Bot: {result['response']}")

    except Exception as e:
        print(f"❌ Failed to start chatbot: {str(e)}")


if __name__ == "__main__":
    main()