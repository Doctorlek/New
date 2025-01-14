const axios = require('axios');

exports.handler = async function(event, context) {
    const body = JSON.parse(event.body);
    const prompt = body.prompt || '';

    if (!prompt) {
        return {
            statusCode: 400,
            body: JSON.stringify({ error: 'Missing prompt' })
        };
    }

    try {
        // כאן נשלחת בקשה ל-API של Hugging Face ללא API Key
        const response = await axios.post(
            'https://api-inference.huggingface.co/models/gpt2', // URL של המודל ב-Hugging Face
            { inputs: prompt }
        );

        return {
            statusCode: 200,
            body: JSON.stringify({ response: response.data[0].generated_text })
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Error with Hugging Face API' })
        };
    }
};
