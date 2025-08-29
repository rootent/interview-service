# RAG Medical Research Assistant

A Retrieval-Augmented Generation (RAG) chatbot that provides accurate answers based on medical research papers. The system prevents hallucination by only using information from the provided documents and includes confidence scoring.

## Features

- **No Hallucination**: Only provides answers based on actual document content
- **Confidence Scoring**: Shows how confident the system is in its answers
- **Source Attribution**: Displays which documents were used for each answer
- **Modern UI**: Beautiful, responsive frontend with real-time chat
- **Error Handling**: Gracefully handles out-of-scope questions
- **CORS Support**: Ready for web deployment

## What's Included

The system comes with two medical research papers:
- **Alzheimer's Disease Prediction** using Machine Learning and Transfer Learning Models
- **ECG Image Analysis** for Medical Issue Detection using Deep Transfer Learning Techniques

## Setup Instructions

### 1. Install Dependencies

**Option A: Simple installation (recommended for Windows)**
```bash
pip install -r requirements-simple.txt
```

**Option B: Version-specific installation**
```bash
pip install -r requirements.txt
```

**If you encounter build errors on Windows, try:**
```bash
# Install Microsoft Visual C++ Build Tools first
# Then use the simple requirements
pip install -r requirements-simple.txt
```

### 2. API Key Setup

You'll need a Google Gemini API key. Get one from [Google AI Studio](https://makersuite.google.com/app/apikey).

**Important**: The current API key in the code is a placeholder. Replace it with your actual key in `main.py`:

```python
genai.configure(api_key="YOUR_ACTUAL_API_KEY_HERE")
```

### 3. Run the Backend

```bash
python main.py
```

The API will start on `http://localhost:8001`

### 4. Open the Frontend

Open `chatbot.html` in your web browser, or serve it using a local server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Then open http://localhost:8000/chatbot.html
```

## API Endpoints

### POST `/chat`
Main chat endpoint for asking questions.

**Request:**
```json
{
  "query": "What is Alzheimer's disease?"
}
```

**Response:**
```json
{
  "answer": "Alzheimer's disease is a progressive neurodegenerative disorder...",
  "confidence": 85.2,
  "sources": ["Alzheimers_Disease_Prediction_using_Machine_Learning_and_Transfer_Learning_Models.pdf"],
  "is_relevant": true
}
```

### GET `/health`
Check system health and document count.

### GET `/`
API information and available endpoints.

## How It Works

1. **Document Processing**: PDFs are loaded, split into chunks, and embedded using sentence transformers
2. **Vector Search**: When a question is asked, the system finds the most relevant document chunks
3. **Relevance Scoring**: Calculates confidence based on semantic similarity
4. **Answer Generation**: Uses Google Gemini to generate answers from relevant context only
5. **Hallucination Prevention**: Strict prompts ensure only document content is used

## Customization

### Adding New Documents

1. Place new PDF files in the `data/` folder
2. Restart the backend
3. The system will automatically index new documents

### Adjusting Confidence Threshold

Modify the `CONFIDENCE_THRESHOLD` in `main.py`:

```python
CONFIDENCE_THRESHOLD = 0.3  # Lower = more permissive, Higher = stricter
```

### Changing the Model

Modify the Gemini model in `main.py`:

```python
model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-pro"
```

## Troubleshooting

### Windows-Specific Issues

1. **Build errors during pip install**:
   - Install Microsoft Visual C++ Build Tools
   - Use `requirements-simple.txt` instead
   - Try installing packages one by one

2. **Memory issues**:
   - Reduce chunk size in the text splitter
   - Use smaller embedding models

3. **PDF loading errors**:
   - Ensure PDFs are not corrupted
   - Try different PDF libraries

### Common Issues

1. **"No documents loaded"**: Check that PDFs are in the `data/` folder
2. **API errors**: Verify your Gemini API key is correct
3. **Frontend not connecting**: Ensure backend is running on port 8001
4. **Memory issues**: Reduce chunk size in the text splitter

### Performance Tips

- Use smaller chunk sizes for faster processing
- Reduce the number of retrieved documents (k parameter)
- Consider using a lighter embedding model

## Security Notes

- The current CORS settings allow all origins (`*`) - restrict this in production
- API keys should be stored in environment variables, not hardcoded
- Consider rate limiting for production use

## License

This project is for educational and research purposes. Please ensure you have proper rights to use any PDF documents you include.

## Contributing

Feel free to submit issues and enhancement requests!
