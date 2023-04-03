import openai
import tiktoken

openai.api_key = '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'
system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit= 4096
conversation=[]
conversation.append(system_message)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def runchatGPT(prompt):
    global conversation
    conversation.append({"role": "user", "content": prompt})
    conv_history_tokens = num_tokens_from_messages(conversation)

    while (conv_history_tokens+max_response_tokens >= token_limit):
        del conversation[1]
        conv_history_tokens = num_tokens_from_messages(conversation)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = conversation,
        temperature=1,
        max_tokens=max_response_tokens,
        top_p=0.9
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    return str(response['choices'][0]['message']['content'])