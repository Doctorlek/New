import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# טוען את המודל והטוקניזר
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_response(input_text):
    try:
        prompt = f"As a friendly and playful dog, here is my response to your statement. Woof! Your statement was: {input_text}. Now, here’s my response:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attention_mask = (input_ids != pad_token_id).to(device)

        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=100,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.9
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        return f"{response} 🐶"
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)} 🐶"

def handler(event, context):
    try:
        body = json.loads(event['body'])
        input_text = body.get("input_text", "")
        
        if not input_text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No input text provided'})
            }
        
        response = generate_response(input_text)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'response': response})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
