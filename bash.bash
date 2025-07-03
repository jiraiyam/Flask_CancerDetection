curl -X POST "https://api.groq.com/openai/v1/chat/completions" \
     -H "Authorization: Bearer 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" \
     -H "Content-Type: application/json" \
     -d '{
          "messages": [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "What is cancer?"}
          ],
          "model": "mistral-saba-24b"
     }'