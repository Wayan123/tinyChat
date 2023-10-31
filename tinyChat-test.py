import transformers
from peft import PeftModel

model_name = "google/flan-t5-large"
peft_model_id = "Leadmatic/tinyChat"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
base_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)

while True:
    user_input = input("User: ")  # Meminta input pertanyaan dari pengguna
    if user_input.lower() == "exit":
        break  # Keluar dari loop jika pengguna memasukkan "exit"
    
    # inputs = tokenizer("translate English to German: " + user_input, return_tensors="pt")
    inputs = tokenizer(user_input, return_tensors="pt")
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = peft_model.generate(**inputs, max_length=300, do_sample=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print("tinychat:", response)  # Menampilkan jawaban dari model
